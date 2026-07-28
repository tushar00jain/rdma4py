# pyre-ignore-all-errors[11]: CuTe scalar objects are runtime annotations.
"""Optional fault-tolerant CUDA all-reduce over RC ibverbs queue pairs.

This module deliberately is not imported by :mod:`ibverbs`. It requires a
PyTorch nightly with the Python ProcessGroup and reconfiguration APIs. CuTe DSL
is loaded lazily when :meth:`ProcessGroup.allreduce` is first called. The
transport uses GPUDirect RDMA only; it never uses CUDA IPC, NVLink, or NCCL.

The implementation is one process per CUDA device.  Each process supplies two
VMM-backed CUDA byte buffers (``work_buffer`` and ``scratch_buffer``), exchanges
the opaque value returned by :meth:`ProcessGroup.get_reconfigure_handle` out of
band, and calls :meth:`ProcessGroup.reconfigure` with the same membership and
fresh UUID.  A deterministic rank-ordered ring is then built with multiple RC
queue pairs.

Only ``SUM`` is currently implemented.  Floating-point addition follows the
NCCL Ring/Simple operation and chunk order.  Bitwise comparison with NCCL also
requires NCCL to use the same ordered ring, Simple protocol, CTA count, and
buffer size; the accompanying benchmark pins the tunable settings and reports
an exact-byte failure if the selected NCCL topology uses a different ring.
"""

from __future__ import annotations

import base64
import ctypes
import json
import math
import os
import select
import socket
import struct
import time
import uuid as uuid_module
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch  # pyre-ignore[21]: Optional nightly dependency.
import torch.distributed as dist  # pyre-ignore[21]: Optional nightly dependency.
from ibverbs import (  # pyre-ignore[21]: Implemented by the Cython extension.
    _ibverbs as _ib,
    cuda as _ibcuda,
)
from ibverbs.enums import AccessFlags
from ibverbs.helpers import connect_rc, local_qp_info, QPInfo

_HANDLE_PREFIX = "ibverbs-allreduce-v1:"
_FRAME = struct.Struct("!I")
_NCCL_CELL_BYTES = 16 * 1024
_NCCL_SIMPLE_CHANNEL_BYTES = 512 * 64
_NCCL_SIMPLE_CHUNK_BYTES = 2 * 1024 * 1024
_GPU_REDUCE_THREADS = 1024
_GPUNETIO_DTYPE_CODES = {
    name: code
    for code, name in enumerate(
        (
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "int8",
            "uint8",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float8_e4m3fn",
            "float8_e5m2",
        )
    )
}
_MAX_FRAME = 16 * 1024 * 1024


def _require_torch_nightly() -> None:
    c10d = torch._C._distributed_c10d
    if not hasattr(c10d, "ReconfigureOptions"):
        raise RuntimeError(
            "fault_tolerant_allreduce requires a PyTorch nightly with c10d "
            "ReconfigureOptions support"
        )


class AllReduceError(RuntimeError):
    """Base class for communicator and collective failures."""


class AllReduceTimeoutError(TimeoutError, AllReduceError):
    """An all-reduce or reconfiguration exceeded its configured deadline."""


class ReconfigureError(AllReduceError):
    """A membership handle or ring reconstruction was invalid."""


class Work(dist.Work):
    """Completed nightly c10d work for synchronous GPUNetIO operations.

    Reconfiguration and all-reduce finish before this object is returned. The
    object still implements the PyProcessGroup trampoline's ``Work`` contract.
    """

    def __init__(self, result: Any = None):
        super().__init__()
        self._result = result
        self._future = torch.futures.Future()
        self._future.set_result(result)

    def is_completed(self) -> bool:
        """Return ``True``; reconfiguration finishes before this is returned."""

        return True

    def is_success(self) -> bool:
        """Return ``True`` because failures are raised synchronously."""

        return True

    def exception(self) -> None:
        """Return ``None`` because failures are raised synchronously."""

        return None

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        """Return ``True`` because the operation has already completed."""

        del timeout
        return True

    def get_future(self) -> torch.futures.Future:
        """Return an already-completed c10d future."""

        return self._future

    def result(self) -> Any:
        """Return the completed operation result."""

        return self._result


@dataclass(frozen=True)
class _Member:
    raw: str
    stable_rank: int
    nonce: str
    host: str
    port: int
    hca: str
    ib_port: int
    gid_index: int
    gid: str
    max_bytes: int
    qps: int
    num_sms: int


@dataclass
class _Lane:
    qp: Any
    cq: Any

    def close(self) -> None:
        self.qp.close()
        self.cq.close()


@dataclass
class _GpuNetIOState:
    """Resources transferred to GPUNetIO for one communicator generation."""

    outgoing: List[Any]
    incoming: List[Any]
    outgoing_pointers: _CudaAllocation
    incoming_pointers: _CudaAllocation
    status: _CudaAllocation
    signal: Any
    signal_mr: Any
    clock_rate_hz: int

    def close(self) -> None:
        error = None
        for handle in reversed(self.outgoing + self.incoming):
            try:
                handle.close()
            except Exception as exc:
                error = error or exc
        for allocation in (
            self.status,
            self.incoming_pointers,
            self.outgoing_pointers,
        ):
            try:
                allocation.close()
            except Exception as exc:
                error = error or exc
        try:
            self.signal_mr.close()
        except Exception as exc:
            error = error or exc
        if error is not None:
            raise error


class _CudaAllocation:
    """Small current-context CUDA allocation used for device control data."""

    _lib = None

    @classmethod
    def _cuda(cls):
        if cls._lib is None:
            lib = ctypes.CDLL("libcuda.so.1")
            lib.cuMemAlloc_v2.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            lib.cuMemAlloc_v2.restype = ctypes.c_int
            lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
            lib.cuMemFree_v2.restype = ctypes.c_int
            lib.cuMemcpyHtoD_v2.argtypes = [
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.cuMemcpyHtoD_v2.restype = ctypes.c_int
            lib.cuMemcpyDtoH_v2.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_size_t,
            ]
            lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
            lib.cuMemsetD8_v2.argtypes = [
                ctypes.c_uint64,
                ctypes.c_ubyte,
                ctypes.c_size_t,
            ]
            lib.cuMemsetD8_v2.restype = ctypes.c_int
            lib.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
            lib.cuCtxGetDevice.restype = ctypes.c_int
            lib.cuDeviceGetAttribute.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.cuDeviceGetAttribute.restype = ctypes.c_int
            cls._lib = lib
        return cls._lib

    @staticmethod
    def _check(function: str, status: int) -> None:
        if status != 0:
            raise RuntimeError("%s failed with CUDA status %d" % (function, status))

    def __init__(self, size: int):
        self.size = int(size)
        if self.size <= 0:
            raise ValueError("CUDA allocation size must be positive")
        pointer = ctypes.c_uint64()
        self._check(
            "cuMemAlloc", self._cuda().cuMemAlloc_v2(ctypes.byref(pointer), self.size)
        )
        self.ptr = pointer.value

    def write_u64(self, values: Sequence[int]) -> None:
        array = (ctypes.c_uint64 * len(values))(*(int(value) for value in values))
        size = ctypes.sizeof(array)
        if size > self.size:
            raise ValueError("control data exceeds CUDA allocation")
        self._check(
            "cuMemcpyHtoD",
            self._cuda().cuMemcpyHtoD_v2(
                self.ptr, ctypes.cast(array, ctypes.c_void_p), size
            ),
        )

    def zero(self) -> None:
        self._check("cuMemsetD8", self._cuda().cuMemsetD8_v2(self.ptr, 0, self.size))

    def read_i32(self, count: int) -> List[int]:
        array = (ctypes.c_int32 * int(count))()
        size = ctypes.sizeof(array)
        if size > self.size:
            raise ValueError("requested status exceeds CUDA allocation")
        self._check(
            "cuMemcpyDtoH",
            self._cuda().cuMemcpyDtoH_v2(
                ctypes.cast(array, ctypes.c_void_p), self.ptr, size
            ),
        )
        return list(array)

    @classmethod
    def clock_rate_hz(cls) -> int:
        # CU_DEVICE_ATTRIBUTE_CLOCK_RATE is in kHz.
        device = ctypes.c_int()
        value = ctypes.c_int()
        cls._check("cuCtxGetDevice", cls._cuda().cuCtxGetDevice(ctypes.byref(device)))
        cls._check(
            "cuDeviceGetAttribute",
            cls._cuda().cuDeviceGetAttribute(ctypes.byref(value), 13, device.value),
        )
        return value.value * 1000

    def close(self) -> None:
        if self.ptr:
            self._check("cuMemFree", self._cuda().cuMemFree_v2(self.ptr))
            self.ptr = 0


def _seconds(value: Optional[Any], default: float) -> float:
    if value is None:
        value = default
    if isinstance(value, timedelta):
        value = value.total_seconds()
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be a finite positive duration")
    return value


def _b64encode(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _encode_handle(payload: Mapping[str, Any]) -> str:
    return _HANDLE_PREFIX + _b64encode(payload)


def _decode_handle(handle: str) -> _Member:
    if not isinstance(handle, str) or not handle.startswith(_HANDLE_PREFIX):
        raise ReconfigureError("invalid ibverbs all-reduce reconfigure handle")
    encoded = handle[len(_HANDLE_PREFIX) :]
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode())
        required = {
            "stable_rank",
            "nonce",
            "host",
            "port",
            "hca",
            "ib_port",
            "gid_index",
            "gid",
            "max_bytes",
            "qps",
            "num_sms",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("unexpected handle fields")
        member = _Member(raw=handle, **value)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReconfigureError("malformed ibverbs all-reduce handle") from exc
    if (
        member.stable_rank < 0
        or not member.nonce
        or not member.host
        or not 0 < member.port < 65536
        or member.ib_port <= 0
        or member.gid_index < 0
        or len(member.gid) != 32
        or member.max_bytes <= 0
        or member.qps <= 0
        or member.num_sms <= 0
    ):
        raise ReconfigureError("invalid values in ibverbs all-reduce handle")
    try:
        bytes.fromhex(member.gid)
    except ValueError as exc:
        raise ReconfigureError("invalid GID in ibverbs all-reduce handle") from exc
    return member


def _qp_info_dict(info: QPInfo) -> Dict[str, Any]:
    return {
        "qp_num": info.qp_num,
        "psn": info.psn,
        "lid": info.lid,
        "gid": bytes(info.gid).hex(),
        "port": info.port,
        "mtu": info.mtu,
    }


def _qp_info_from_dict(value: Mapping[str, Any]) -> QPInfo:
    try:
        return QPInfo(
            qp_num=int(value["qp_num"]),
            psn=int(value["psn"]),
            lid=int(value["lid"]),
            gid=bytes.fromhex(value["gid"]),
            port=int(value["port"]),
            mtu=int(value["mtu"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconfigureError("invalid QP information from peer") from exc


def _send_frame(sock: socket.socket, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > _MAX_FRAME:
        raise ReconfigureError("reconfigure message is too large")
    sock.sendall(_FRAME.pack(len(payload)) + payload)


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    parts = []
    remaining = count
    while remaining:
        part = sock.recv(remaining)
        if not part:
            raise ReconfigureError("peer closed the reconfigure connection")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def _recv_frame(sock: socket.socket) -> Dict[str, Any]:
    size = _FRAME.unpack(_recv_exact(sock, _FRAME.size))[0]
    if size > _MAX_FRAME:
        raise ReconfigureError("reconfigure message is too large")
    try:
        value = json.loads(_recv_exact(sock, size).decode())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReconfigureError("peer sent malformed reconfigure data") from exc
    if not isinstance(value, dict):
        raise ReconfigureError("peer reconfigure data must be an object")
    return value


def _connect_until(host: str, port: int, deadline: float) -> socket.socket:
    last_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AllReduceTimeoutError(
                "timed out connecting to reconfigure peer %s:%d" % (host, port)
            ) from last_error
        try:
            sock = socket.create_connection((host, port), timeout=min(remaining, 1.0))
            sock.settimeout(remaining)
            return sock
        except OSError as exc:
            last_error = exc
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))


def _accept_member(
    listener: socket.socket,
    expected_handle: str,
    reconfigure_uuid: int,
    deadline: float,
) -> Tuple[socket.socket, Dict[str, Any]]:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AllReduceTimeoutError("timed out accepting the previous ring peer")
        ready, _, _ = select.select([listener], [], [], remaining)
        if not ready:
            raise AllReduceTimeoutError("timed out accepting the previous ring peer")
        sock, _ = listener.accept()
        sock.settimeout(max(0.001, deadline - time.monotonic()))
        try:
            message = _recv_frame(sock)
            if (
                message.get("uuid") == reconfigure_uuid
                and message.get("handle") == expected_handle
            ):
                return sock, message
        except (OSError, AllReduceError):
            pass
        sock.close()


def _check_deadline(deadline: float, what: str = "all-reduce") -> None:
    if time.monotonic() >= deadline:
        raise AllReduceTimeoutError("%s exceeded its timeout" % what)


def _ibv_timeout_exponent(timeout: float, retries: int) -> int:
    # RC local ACK timeout is 4.096 us * 2**exponent.  Leave enough time for
    # every configured retry to finish before the Python collective deadline.
    per_attempt = max(4.096e-6, timeout / (retries + 1))
    return max(0, min(31, int(math.floor(math.log2(per_attempt / 4.096e-6)))))


def _tensor_nbytes(tensor: Any) -> int:
    if not hasattr(tensor, "data_ptr"):
        raise TypeError("expected a CUDA tensor exposing data_ptr()")
    contiguous = getattr(tensor, "is_contiguous", None)
    if callable(contiguous) and not contiguous():
        raise ValueError("all-reduce tensor must be contiguous")
    if not bool(getattr(tensor, "is_cuda", True)):
        raise ValueError("all-reduce tensor must reside on CUDA")
    if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
        return int(tensor.numel()) * int(tensor.element_size())
    if hasattr(tensor, "nbytes"):
        return int(tensor.nbytes)
    raise TypeError("cannot determine tensor byte length")


def _dtype_key(tensor: Any) -> Tuple[str, int]:
    itemsize = int(tensor.element_size())
    name = str(getattr(tensor, "dtype", "")).lower()
    aliases = {
        "torch.float16": "float16",
        "float16": "float16",
        "half": "float16",
        "torch.bfloat16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch.float32": "float32",
        "float32": "float32",
        "float": "float32",
        "torch.float64": "float64",
        "float64": "float64",
        "double": "float64",
        "torch.float8_e4m3fn": "float8_e4m3fn",
        "float8_e4m3fn": "float8_e4m3fn",
        "torch.float8_e5m2": "float8_e5m2",
        "float8_e5m2": "float8_e5m2",
        "torch.int8": "int8",
        "int8": "int8",
        "torch.uint8": "uint8",
        "uint8": "uint8",
        "torch.int32": "int32",
        "int32": "int32",
        "torch.uint32": "uint32",
        "uint32": "uint32",
        "torch.int64": "int64",
        "int64": "int64",
        "torch.uint64": "uint64",
        "uint64": "uint64",
    }
    try:
        key = aliases[name]
    except KeyError as exc:
        raise TypeError("unsupported all-reduce dtype %r" % name) from exc
    widths = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
        "int8": 1,
        "uint8": 1,
        "int32": 4,
        "uint32": 4,
        "int64": 8,
        "uint64": 8,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
    }
    if itemsize != widths[key]:
        raise TypeError("tensor dtype and element_size() disagree")
    return key, itemsize


def _nccl_channel_layout(
    count: int, itemsize: int, num_sms: int
) -> Tuple[int, int, int]:
    """Return ``(cell_elements, cells_per_channel, active_channels)``.

    This is the single-collective case of NCCL's continuous-byte-distribution
    scheduler for Ring/Simple all-reduce.
    """

    cell_elements = max(1, _NCCL_CELL_BYTES // itemsize)
    cells = (count + cell_elements - 1) // cell_elements
    # NCCL 2.29 Ring/Simple starts with 512 worker threads and a 64-byte
    # per-thread threshold, then removes channels while
    # nBytes < nChannels * 512 * 64. The floor is intentional at exact
    # boundaries and determines the rank that owns each reduction chunk.
    nbytes = count * itemsize
    active = min(num_sms, max(1, nbytes // _NCCL_SIMPLE_CHANNEL_BYTES))
    per_channel = (cells + active - 1) // active
    return cell_elements, per_channel, active


class _CuteCopyKernels:
    """Lazily JIT-compiled CuTe staging-copy kernels."""

    _CUTLASS_TYPES = {
        "float16": "Float16",
        "bfloat16": "BFloat16",
        "float32": "Float32",
        "float64": "Float64",
        "float8_e4m3fn": "Float8E4M3FN",
        "float8_e5m2": "Float8E5M2",
        "int8": "Int8",
        "uint8": "Uint8",
        "int32": "Int32",
        "uint32": "Uint32",
        "int64": "Int64",
        "uint64": "Uint64",
    }

    def __init__(self, num_sms: int):
        self.num_sms = num_sms
        self._compiled: Dict[str, Tuple[Any, Any, Any, Any]] = {}

    def _get(self, dtype_key: str, pointer: int) -> Tuple[Any, Any, Any, Any]:
        cached = self._compiled.get(dtype_key)
        if cached is not None:
            return cached
        try:
            import cutlass  # pyre-ignore[21]: Optional dependency.
            import cutlass.cute as cute  # pyre-ignore[21]: Optional dependency.
            from cutlass.cute.runtime import (  # pyre-ignore[21]: Optional dependency.
                make_ptr,
            )
        except ImportError as exc:
            raise RuntimeError(
                "CuTe all-reduce requires Python 3.10+ and the optional "
                "nvidia-cutlass-dsl package; install with "
                "pip install -e './ibverbs[gpunetio-cutedsl]'"
            ) from exc

        # ``from __future__ import annotations`` stores the nested JIT
        # function annotations as strings.  CuTe asks ``inspect`` to resolve
        # them later, which uses module globals rather than this method's
        # closure.
        globals()["cutlass"] = cutlass
        globals()["cute"] = cute

        dtype = getattr(cutlass, self._CUTLASS_TYPES[dtype_key])
        sms = self.num_sms

        @cute.kernel
        def copy_kernel(src: cute.Tensor, dst: cute.Tensor, count: cutlass.Int32):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            index = cutlass.Int32(bid * 256 + tid)
            stride = cutlass.Int32(sms * 256)
            for i in cutlass.range(index, count, stride):
                if i < count:
                    dst[i] = src[i]

        @cute.jit
        def copy_host(
            src_ptr: cute.Pointer, dst_ptr: cute.Pointer, count: cutlass.Int32
        ):
            src = cute.make_tensor(src_ptr, cute.make_layout((count,)))
            dst = cute.make_tensor(dst_ptr, cute.make_layout((count,)))
            copy_kernel(src, dst, count).launch(grid=[sms, 1, 1], block=[256, 1, 1])

        ptr = make_ptr(dtype, pointer, cute.AddressSpace.gmem, assumed_align=16)
        copy = cute.compile(copy_host, ptr, ptr, cutlass.Int32(1))
        cached = (copy, make_ptr, dtype, cutlass)
        self._compiled[dtype_key] = cached
        return cached

    def copy(self, source: int, destination: int, count: int, dtype_key: str) -> None:
        """Copy ``count`` elements with a fixed ``num_sms`` CTA launch."""

        copy, make_ptr, dtype, cutlass = self._get(dtype_key, source)
        src = make_ptr(dtype, source, self._address_space(make_ptr), assumed_align=16)
        dst = make_ptr(
            dtype, destination, self._address_space(make_ptr), assumed_align=16
        )
        copy(src, dst, cutlass.Int32(count))

    @staticmethod
    def _address_space(make_ptr: Any) -> Any:
        # ``make_ptr`` does not expose its enum.  It lives in the parent cute
        # module and is stable across CuTe DSL 4.x.
        import cutlass.cute as cute

        return cute.AddressSpace.gmem


class _CuteGpuNetIOKernels:
    """Persistent CuTe ring kernels whose verbs data path runs on the GPU."""

    def __init__(
        self,
        num_sms: int,
        qps: int,
        *,
        arch: str = "sm_90",
        bitcode=None,
    ):
        self.num_sms = int(num_sms)
        self.qps = int(qps)
        self.arch = arch
        self.bitcode = bitcode
        self._compiled: Dict[str, Tuple[Any, Any, Any, Any]] = {}

    def _get(
        self,
        dtype_key: str,
        data_pointer: int,
        qp_pointer: int,
        status_pointer: int,
    ) -> Tuple[Any, Any, Any, Any]:
        cached = self._compiled.get(dtype_key)
        if cached is not None:
            return cached
        try:
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import make_ptr
            from ibverbs.gpunetio import build_bitcode
            from ibverbs.gpunetio.cutedsl import bind
        except ImportError as exc:
            raise RuntimeError(
                "GPUNetIO all-reduce requires nvidia-cutlass-dsl and the "
                "ibverbs GPUNetIO optional dependencies"
            ) from exc

        bitcode = self.bitcode
        if bitcode is None:
            try:
                gda = bind(arch=self.arch)
            except FileNotFoundError:
                bitcode = build_bitcode(arch=self.arch)
                gda = bind(bitcode, arch=self.arch)
        else:
            gda = bind(bitcode, arch=self.arch)

        globals()["cutlass"] = cutlass
        globals()["cute"] = cute
        dtype = getattr(cutlass, _CuteCopyKernels._CUTLASS_TYPES[dtype_key])
        sms = self.num_sms
        qps = self.qps
        itemsize = dtype.width // 8
        dtype_code = _GPUNETIO_DTYPE_CODES[dtype_key]

        @cute.kernel
        def ring_kernel(
            outgoing_qps: cute.Tensor,
            incoming_qps: cute.Tensor,
            status: cute.Tensor,
            work: cute.Tensor,
            scratch: cute.Tensor,
            count: cutlass.Int32,
            rank: cutlass.Int32,
            world: cutlass.Int32,
            remote_work_addr: cutlass.Uint64,
            remote_work_rkey: cutlass.Uint32,
            remote_scratch_addr: cutlass.Uint64,
            remote_scratch_rkey: cutlass.Uint32,
            work_addr: cutlass.Uint64,
            work_lkey: cutlass.Uint32,
            scratch_addr: cutlass.Uint64,
            signal_addr: cutlass.Uint64,
            signal_lkey: cutlass.Uint32,
            timeout_cycles: cutlass.Uint64,
            cell_elements: cutlass.Int32,
            cells_per_channel: cutlass.Int32,
            active_channels: cutlass.Int32,
            base_chunk: cutlass.Int32,
            align_elements: cutlass.Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            channel, _, _ = cute.arch.block_idx()
            if channel < active_channels:
                # A channel owns its QP. Sharing an RC receive queue across
                # independently scheduled CTAs would let untagged SEND
                # notifications match a different channel's receive.
                lane = channel
                outgoing_qp = cutlass.Uint64(outgoing_qps[lane])
                incoming_qp = cutlass.Uint64(incoming_qps[lane])
                deadline = cutlass.Uint64(0)
                if tid == 0:
                    status[channel] = cutlass.Int32(0)
                    deadline = cutlass.Uint64(gda.clock64() + timeout_cycles)
                cute.arch.sync_threads()

                channel_start = (
                    cutlass.Int32(channel) * cells_per_channel * cell_elements
                )
                channel_count = min(
                    count - channel_start, cells_per_channel * cell_elements
                )
                loop_count = world * base_chunk
                for elem_offset in cutlass.range(0, channel_count, loop_count):
                    remaining = channel_count - elem_offset
                    chunk = base_chunk
                    if remaining < loop_count:
                        chunk = (remaining + world - 1) // world
                        chunk = (
                            (chunk + align_elements - 1) // align_elements
                        ) * align_elements

                    # Reduce-scatter. Each CTA posts its receive before an
                    # RDMA Write with Immediate; the remote CQE is also the
                    # GPU-memory visibility notification. Empty chunks use a
                    # one-byte SEND so every rank still advances the ring.
                    for step in cutlass.range(1, world):
                        send_owner = (rank - step + world) % world
                        recv_owner = (rank - step - 1 + 2 * world) % world
                        send_local = send_owner * chunk
                        recv_local = recv_owner * chunk
                        send_length = min(
                            chunk, max(cutlass.Int32(0), remaining - send_local)
                        )
                        recv_length = min(
                            chunk, max(cutlass.Int32(0), remaining - recv_local)
                        )
                        send_element = channel_start + elem_offset + send_local
                        recv_element = channel_start + elem_offset + recv_local
                        if tid == 0:
                            if status[channel] == 0:
                                recv_ticket = gda.recv_exclusive(
                                    incoming_qp,
                                    signal_addr,
                                    signal_lkey,
                                    cutlass.Uint64(1),
                                )
                                send_ticket = cutlass.Uint64(0)
                                if send_length > 0:
                                    send_ticket = gda.put_imm_exclusive(
                                        outgoing_qp,
                                        remote_scratch_addr
                                        + cutlass.Uint64(send_element * itemsize),
                                        remote_scratch_rkey,
                                        work_addr
                                        + cutlass.Uint64(send_element * itemsize),
                                        work_lkey,
                                        cutlass.Uint64(send_length * itemsize),
                                        cutlass.Uint32(channel),
                                    )
                                else:
                                    send_ticket = gda.send_exclusive(
                                        outgoing_qp,
                                        signal_addr,
                                        signal_lkey,
                                        cutlass.Uint64(1),
                                    )
                                if status[channel] == 0:
                                    operation_status = gda.wait_send_until_exclusive(
                                        outgoing_qp, send_ticket, deadline
                                    )
                                    if operation_status != 0:
                                        status[channel] = operation_status
                                if status[channel] == 0:
                                    operation_status = gda.wait_recv_until_exclusive(
                                        incoming_qp, recv_ticket, deadline
                                    )
                                    if operation_status != 0:
                                        status[channel] = operation_status
                        cute.arch.sync_threads()
                        if status[channel] == 0:
                            fence_status = gda.fence_acquire()
                            if fence_status != 0:
                                status[channel] = fence_status
                            reduce_status = gda.reduce_volatile(
                                work_addr + cutlass.Uint64(recv_element * itemsize),
                                scratch_addr + cutlass.Uint64(recv_element * itemsize),
                                cutlass.Uint32(tid),
                                cutlass.Uint32(recv_length),
                                cutlass.Uint32(_GPU_REDUCE_THREADS),
                                cutlass.Int32(dtype_code),
                            )
                            if reduce_status != 0:
                                status[channel] = reduce_status
                        cute.arch.sync_threads()

                    # All-gather writes directly into the next rank's work
                    # buffer.  The next iteration can immediately forward it.
                    for step in cutlass.range(0, world - 1):
                        send_owner = (rank - step + world) % world
                        send_local = send_owner * chunk
                        send_length = min(
                            chunk, max(cutlass.Int32(0), remaining - send_local)
                        )
                        send_element = channel_start + elem_offset + send_local
                        if tid == 0:
                            if status[channel] == 0:
                                recv_ticket = gda.recv_exclusive(
                                    incoming_qp,
                                    signal_addr,
                                    signal_lkey,
                                    cutlass.Uint64(1),
                                )
                                send_ticket = cutlass.Uint64(0)
                                if send_length > 0:
                                    send_ticket = gda.put_imm_exclusive(
                                        outgoing_qp,
                                        remote_work_addr
                                        + cutlass.Uint64(send_element * itemsize),
                                        remote_work_rkey,
                                        work_addr
                                        + cutlass.Uint64(send_element * itemsize),
                                        work_lkey,
                                        cutlass.Uint64(send_length * itemsize),
                                        cutlass.Uint32(channel),
                                    )
                                else:
                                    send_ticket = gda.send_exclusive(
                                        outgoing_qp,
                                        signal_addr,
                                        signal_lkey,
                                        cutlass.Uint64(1),
                                    )
                                if status[channel] == 0:
                                    operation_status = gda.wait_send_until_exclusive(
                                        outgoing_qp, send_ticket, deadline
                                    )
                                    if operation_status != 0:
                                        status[channel] = operation_status
                                if status[channel] == 0:
                                    operation_status = gda.wait_recv_until_exclusive(
                                        incoming_qp, recv_ticket, deadline
                                    )
                                    if operation_status != 0:
                                        status[channel] = operation_status
                        cute.arch.sync_threads()

        @cute.jit
        def ring_host(
            outgoing_qps_ptr: cute.Pointer,
            incoming_qps_ptr: cute.Pointer,
            status_ptr: cute.Pointer,
            work_ptr: cute.Pointer,
            scratch_ptr: cute.Pointer,
            count: cutlass.Int32,
            rank: cutlass.Int32,
            world: cutlass.Int32,
            remote_work_addr: cutlass.Uint64,
            remote_work_rkey: cutlass.Uint32,
            remote_scratch_addr: cutlass.Uint64,
            remote_scratch_rkey: cutlass.Uint32,
            work_addr: cutlass.Uint64,
            work_lkey: cutlass.Uint32,
            scratch_addr: cutlass.Uint64,
            signal_addr: cutlass.Uint64,
            signal_lkey: cutlass.Uint32,
            timeout_cycles: cutlass.Uint64,
            cell_elements: cutlass.Int32,
            cells_per_channel: cutlass.Int32,
            active_channels: cutlass.Int32,
            base_chunk: cutlass.Int32,
            align_elements: cutlass.Int32,
        ):
            outgoing_qps = cute.make_tensor(outgoing_qps_ptr, cute.make_layout((qps,)))
            incoming_qps = cute.make_tensor(incoming_qps_ptr, cute.make_layout((qps,)))
            status = cute.make_tensor(status_ptr, cute.make_layout((sms,)))
            work = cute.make_tensor(work_ptr, cute.make_layout((count,)))
            scratch = cute.make_tensor(scratch_ptr, cute.make_layout((count,)))
            ring_kernel(
                outgoing_qps,
                incoming_qps,
                status,
                work,
                scratch,
                count,
                rank,
                world,
                remote_work_addr,
                remote_work_rkey,
                remote_scratch_addr,
                remote_scratch_rkey,
                work_addr,
                work_lkey,
                scratch_addr,
                signal_addr,
                signal_lkey,
                timeout_cycles,
                cell_elements,
                cells_per_channel,
                active_channels,
                base_chunk,
                align_elements,
            ).launch(grid=[sms, 1, 1], block=[_GPU_REDUCE_THREADS, 1, 1])

        qp_ptr = make_ptr(
            cutlass.Uint64, qp_pointer, cute.AddressSpace.gmem, assumed_align=8
        )
        status_ptr = make_ptr(
            cutlass.Int32, status_pointer, cute.AddressSpace.gmem, assumed_align=4
        )
        data_ptr = make_ptr(
            dtype, data_pointer, cute.AddressSpace.gmem, assumed_align=16
        )
        compiled = cute.compile(
            ring_host,
            qp_ptr,
            qp_ptr,
            status_ptr,
            data_ptr,
            data_ptr,
            cutlass.Int32(1),
            cutlass.Int32(0),
            cutlass.Int32(2),
            cutlass.Uint64(0),
            cutlass.Uint64(0),
            cutlass.Uint32(0),
            cutlass.Uint64(0),
            cutlass.Uint32(0),
            cutlass.Uint64(0),
            cutlass.Uint32(0),
            cutlass.Uint64(0),
            cutlass.Uint32(0),
            cutlass.Uint64(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
        )
        cached = (compiled, make_ptr, dtype, cutlass)
        self._compiled[dtype_key] = cached
        return cached

    def allreduce(
        self,
        *,
        outgoing_qps: int,
        incoming_qps: int,
        status: int,
        work: int,
        scratch: int,
        count: int,
        dtype_key: str,
        itemsize: int,
        rank: int,
        world: int,
        remote_buffers: Mapping[str, int],
        work_lkey: int,
        signal_addr: int,
        signal_lkey: int,
        timeout_cycles: int,
    ) -> None:
        compiled, make_ptr, dtype, cutlass = self._get(
            dtype_key, work, outgoing_qps, status
        )
        import cutlass.cute as cute

        qp_out = make_ptr(
            cutlass.Uint64, outgoing_qps, cute.AddressSpace.gmem, assumed_align=8
        )
        qp_in = make_ptr(
            cutlass.Uint64, incoming_qps, cute.AddressSpace.gmem, assumed_align=8
        )
        status_ptr = make_ptr(
            cutlass.Int32, status, cute.AddressSpace.gmem, assumed_align=4
        )
        work_ptr = make_ptr(dtype, work, cute.AddressSpace.gmem, assumed_align=16)
        scratch_ptr = make_ptr(dtype, scratch, cute.AddressSpace.gmem, assumed_align=16)
        cell, cells_per_channel, active = _nccl_channel_layout(
            count, itemsize, self.num_sms
        )
        compiled(
            qp_out,
            qp_in,
            status_ptr,
            work_ptr,
            scratch_ptr,
            cutlass.Int32(count),
            cutlass.Int32(rank),
            cutlass.Int32(world),
            cutlass.Uint64(remote_buffers["work_addr"]),
            cutlass.Uint32(remote_buffers["work_rkey"]),
            cutlass.Uint64(remote_buffers["scratch_addr"]),
            cutlass.Uint32(remote_buffers["scratch_rkey"]),
            cutlass.Uint64(work),
            cutlass.Uint32(work_lkey),
            cutlass.Uint64(scratch),
            cutlass.Uint64(signal_addr),
            cutlass.Uint32(signal_lkey),
            cutlass.Uint64(timeout_cycles),
            cutlass.Int32(cell),
            cutlass.Int32(cells_per_channel),
            cutlass.Int32(active),
            cutlass.Int32(max(1, _NCCL_SIMPLE_CHUNK_BYTES // itemsize)),
            cutlass.Int32(max(1, 16 // itemsize)),
        )


def _timeout_value(value: Any) -> Optional[timedelta]:
    if not isinstance(value, timedelta) or value.total_seconds() <= 0:
        return None
    return value


def _find_device(name: str) -> Any:
    for device in _ib.get_device_list():  # pyre-ignore[16]: Cython export.
        if device.name == name:
            return device
    raise RuntimeError("RDMA device %r was not found" % name)


def _find_gid_index(context: Any, device_name: str, port: int) -> int:
    base = "/sys/class/infiniband/%s/ports/%d/gid_attrs/types" % (
        device_name,
        port,
    )
    best = None
    attributes = context.query_port(port)
    for index in range(attributes.gid_tbl_len):
        gid = context.query_gid(port, index)
        if gid.raw == b"\0" * 16:
            continue
        try:
            with open(os.path.join(base, str(index))) as stream:
                gid_type = stream.read().strip().lower()
        except OSError:
            gid_type = ""
        link_local = gid.raw[0] == 0xFE and (gid.raw[1] & 0xC0) == 0x80
        score = (4 if "v2" in gid_type else 0) + (0 if link_local else 2)
        if best is None or score > best[0]:
            best = (score, index)
    if best is None:
        raise RuntimeError("no usable GID on %s port %d" % (device_name, port))
    return best[1]


@dataclass
class ProcessGroupOptions:
    """Resources and GPUNetIO tuning for the nightly c10d backend."""

    hca: str
    max_bytes: int = 64 * 1024 * 1024
    gpu: Optional[int] = None
    gid_index: Optional[int] = None
    port: int = 1
    num_sms: int = 32
    qps: Optional[int] = None
    queue_depth: int = 256
    advertise_host: Optional[str] = None
    bind_host: str = "0.0.0.0"
    rendezvous_port: int = 0
    gpunetio_arch: str = "sm_90"
    gpunetio_bitcode: Any = None
    nic_handler: str = "gpu"
    stable_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.hca:
            raise ValueError("hca must name the GPU-local RDMA device")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")


class ProcessGroup(dist.ProcessGroup):
    """Nightly PyProcessGroup implementing rank-ordered GPUNetIO all-reduce.

    Args:
        store: c10d store retained by the nightly PyProcessGroup trampoline.
        rank: Initial c10d group rank.
        size: Initial c10d group size.
        context: Open :class:`ibverbs.Context` for the GPU-local HCA.
        pd: Protection domain allocated from ``context``.
        work_buffer: Contiguous VMM-backed CUDA byte tensor used internally.
        scratch_buffer: A distinct buffer with at least the same capacity.
        stable_rank: Stable application rank encoded in reconfigure handles;
            defaults to ``rank``.
        gid_index: Local GID index used to connect RC QPs.
        port: Active HCA port number.
        advertise_host: Address peers can use for the reconfigure rendezvous.
        bind_host: Local address for the rendezvous listener.
        rendezvous_port: Listener port, or zero for an ephemeral port.
        qps: RC queue pairs used in parallel for each ring direction. The
            default uses one QP per CuTe channel.
        num_sms: CuTe CTA count.  The benchmark forces NCCL to this same count.
        timeout: Default all-reduce and reconfiguration deadline.
        queue_depth: Maximum posted writes per QP in one ring round.
        gpu: CUDA device index or PCI address exported through GPUNetIO. The
            default selects the current CUDA context.
        gpunetio_arch: CUDA architecture passed to the GPUNetIO bitcode build.
        gpunetio_bitcode: Optional precompiled GPUNetIO device bitcode path.
        nic_handler: GPUNetIO doorbell handler. ``"gpu"`` rejects CPU proxy
            fallback and is the only mode intended for benchmark results.

    ``work_buffer`` and ``scratch_buffer`` are retained and registered for the
    lifetime of the group.  They must be allocated on the current CUDA device;
    torch users must enable ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
    before importing torch.
    """

    backend_name = "ibverbs"

    def __init__(
        self,
        store: dist.Store,
        rank: int,
        size: int,
        context: Any,
        pd: Any,
        work_buffer: Any,
        scratch_buffer: Any,
        *,
        stable_rank: Optional[int] = None,
        gid_index: int,
        port: int = 1,
        advertise_host: Optional[str] = None,
        bind_host: str = "0.0.0.0",
        rendezvous_port: int = 0,
        qps: Optional[int] = None,
        num_sms: int = 32,
        timeout: Any = 30.0,
        queue_depth: int = 256,
        gpu: Any = None,
        gpunetio_arch: str = "sm_90",
        gpunetio_bitcode: Any = None,
        nic_handler: str = "gpu",
    ):
        _require_torch_nightly()
        super().__init__(store, rank, size)  # pyre-ignore[19]
        self._store = store
        self._initial_rank = int(rank)
        self._initial_size = int(size)
        if stable_rank is None:
            stable_rank = rank
        num_sms = int(num_sms)
        if qps is None:
            qps = num_sms
        if int(stable_rank) < 0:
            raise ValueError("stable_rank must be non-negative")
        if int(qps) <= 0 or num_sms <= 0 or int(queue_depth) <= 0:
            raise ValueError("qps, num_sms, and queue_depth must be positive")
        if int(qps) < num_sms:
            raise ValueError(
                "GPUNetIO requires at least one QP per SM/channel; "
                "set qps >= num_sms"
            )
        if nic_handler not in {"gpu", "auto", "cpu"}:
            raise ValueError("nic_handler must be 'gpu', 'auto', or 'cpu'")
        work_bytes = _tensor_nbytes(work_buffer)
        scratch_bytes = _tensor_nbytes(scratch_buffer)
        if work_bytes <= 0 or scratch_bytes < work_bytes:
            raise ValueError("scratch_buffer must be at least as large as work_buffer")
        if int(work_buffer.data_ptr()) % 16 or int(scratch_buffer.data_ptr()) % 16:
            raise ValueError("staging buffers must be at least 16-byte aligned")

        self.context = context
        self.pd = pd
        self.work_buffer = work_buffer
        self.scratch_buffer = scratch_buffer
        self.stable_rank = int(stable_rank)
        self.gid_index = int(gid_index)
        self.port = int(port)
        self.qps = int(qps)
        self.num_sms = num_sms
        self.timeout = _seconds(timeout, 30.0)
        self.queue_depth = int(queue_depth)
        self.gpu = gpu
        self.gpunetio_arch = str(gpunetio_arch)
        self.gpunetio_bitcode = gpunetio_bitcode
        self.nic_handler = nic_handler
        self.max_bytes = work_bytes
        self._rank = -1
        self._size = 0
        self.uuid: Optional[int] = None
        self._failed = False
        self._closed = False
        self._outgoing: List[_Lane] = []
        self._incoming: List[_Lane] = []
        self._next_buffers: Optional[Dict[str, int]] = None
        self._gpunetio: Optional[_GpuNetIOState] = None
        self._used_uuids = set()
        self._kernels = _CuteCopyKernels(self.num_sms)
        self._gpunetio_kernels = _CuteGpuNetIOKernels(
            self.num_sms,
            self.qps,
            arch=self.gpunetio_arch,
            bitcode=self.gpunetio_bitcode,
        )

        access = AccessFlags.LOCAL_WRITE | AccessFlags.REMOTE_WRITE
        self._work_mr = _ibcuda.register_tensor(pd, work_buffer, access)
        try:
            self._scratch_mr = _ibcuda.register_tensor(pd, scratch_buffer, access)
        except Exception:
            self._work_mr.close()
            raise

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._listener.bind((bind_host, int(rendezvous_port)))
            self._listener.listen(16)
        except Exception:
            self._scratch_mr.close()
            self._work_mr.close()
            self._listener.close()
            raise
        bound_port = self._listener.getsockname()[1]
        host = advertise_host or socket.getfqdn()
        gid = context.query_gid(self.port, self.gid_index)
        self._handle = _encode_handle(
            {
                "stable_rank": self.stable_rank,
                "nonce": uuid_module.uuid4().hex,
                "host": host,
                "port": bound_port,
                "hca": context.name,
                "ib_port": self.port,
                "gid_index": self.gid_index,
                "gid": bytes(gid.raw).hex(),
                "max_bytes": self.max_bytes,
                "qps": self.qps,
                "num_sms": self.num_sms,
            }
        )

    @classmethod
    def create(
        cls,
        store: dist.Store,
        rank: int,
        size: int,
        options: ProcessGroupOptions,
        timeout: timedelta,
        *,
        initialize: bool,
        group_id: str,
    ) -> "ProcessGroup":
        """Allocate CUDA/RDMA resources and optionally form the initial ring."""

        _require_torch_nightly()
        gpu = torch.cuda.current_device() if options.gpu is None else int(options.gpu)
        torch.cuda.set_device(gpu)
        device = _find_device(options.hca)
        context = device.open()
        pd = context.alloc_pd()
        try:
            gid_index = options.gid_index
            if gid_index is None:
                gid_index = _find_gid_index(context, options.hca, options.port)
            work = torch.empty(options.max_bytes, dtype=torch.uint8, device=gpu)
            scratch = torch.empty(options.max_bytes, dtype=torch.uint8, device=gpu)
            group = cls(
                store,
                rank,
                size,
                context,
                pd,
                work,
                scratch,
                stable_rank=options.stable_rank,
                gid_index=gid_index,
                port=options.port,
                advertise_host=options.advertise_host,
                bind_host=options.bind_host,
                rendezvous_port=options.rendezvous_port,
                qps=options.qps,
                num_sms=options.num_sms,
                timeout=timeout,
                queue_depth=options.queue_depth,
                gpu=gpu,
                gpunetio_arch=options.gpunetio_arch,
                gpunetio_bitcode=options.gpunetio_bitcode,
                nic_handler=options.nic_handler,
            )
        except Exception:
            pd.close()
            context.close()
            raise
        try:
            if initialize:
                group._initialize(group_id, timeout)
        except Exception:
            group.close()
            raise
        return group

    def _initialize(self, group_id: str, timeout: timedelta) -> None:
        key_prefix = "ibverbs-initial-%s" % (group_id or "default")
        self._store.set(
            "%s/%d" % (key_prefix, self._initial_rank),
            self.get_reconfigure_handle(),
        )
        handles = [
            self._store.get("%s/%d" % (key_prefix, rank)).decode("utf-8")
            for rank in range(self._initial_size)
        ]
        self._reconfigure(0, handles, timeout=timeout).wait()

    @property
    def supports_reconfigure(self) -> bool:
        """Return ``True`` for nightly c10d fault-tolerance discovery."""

        return True

    def get_reconfigure_handle(self) -> str:
        """Return this rank's opaque, out-of-band membership handle."""

        if self._closed:
            raise ReconfigureError("process group is closed")
        return self._handle

    def _create_lanes(self) -> Tuple[List[_Lane], List[_Lane]]:
        outgoing = []
        incoming = []
        try:
            for _ in range(self.qps):
                for collection in (outgoing, incoming):
                    cq = self.context.create_cq(self.queue_depth + 8)
                    try:
                        qp = self.pd.create_qp(
                            _ib.QPInitAttr(  # pyre-ignore[16]: Cython export.
                                send_cq=cq,
                                recv_cq=cq,
                                max_send_wr=self.queue_depth,
                                max_recv_wr=max(
                                    8, (self.num_sms + self.qps - 1) // self.qps + 2
                                ),
                            )
                        )
                    except Exception:
                        cq.close()
                        raise
                    collection.append(_Lane(qp, cq))
        except Exception:
            for lane in reversed(outgoing + incoming):
                lane.close()
            raise
        return outgoing, incoming

    def _lane_infos(self, lanes: Sequence[_Lane]) -> List[Dict[str, Any]]:
        port_attr = self.context.query_port(self.port)
        gid = self.context.query_gid(self.port, self.gid_index)
        return [
            _qp_info_dict(local_qp_info(lane.qp, port_attr, gid, port=self.port, psn=0))
            for lane in lanes
        ]

    def _buffer_info(self) -> Dict[str, int]:
        return {
            "work_addr": self._work_mr.addr,
            "work_rkey": self._work_mr.rkey,
            "scratch_addr": self._scratch_mr.addr,
            "scratch_rkey": self._scratch_mr.rkey,
        }

    def _export_gpunetio(
        self, outgoing: Sequence[_Lane], incoming: Sequence[_Lane]
    ) -> _GpuNetIOState:
        """Transfer freshly connected QPs to the direct GPU data path."""

        from ibverbs.gpunetio import DeviceQP

        device_outgoing = []
        device_incoming = []
        allocations = []
        signal_mr = None
        try:
            for lane in outgoing:
                device_outgoing.append(
                    DeviceQP.export(lane.qp, gpu=self.gpu, nic_handler=self.nic_handler)
                )
            for lane in incoming:
                device_incoming.append(
                    DeviceQP.export(lane.qp, gpu=self.gpu, nic_handler=self.nic_handler)
                )
            outgoing_pointers = _CudaAllocation(8 * self.qps)
            allocations.append(outgoing_pointers)
            incoming_pointers = _CudaAllocation(8 * self.qps)
            allocations.append(incoming_pointers)
            status = _CudaAllocation(4 * self.num_sms)
            allocations.append(status)
            outgoing_pointers.write_u64(
                [handle.device_ptr for handle in device_outgoing]
            )
            incoming_pointers.write_u64(
                [handle.device_ptr for handle in device_incoming]
            )
            status.zero()
            signal = ctypes.create_string_buffer(1)
            signal_mr = self.pd.reg_mr(
                ctypes.addressof(signal), 1, AccessFlags.LOCAL_WRITE
            )
            return _GpuNetIOState(
                outgoing=device_outgoing,
                incoming=device_incoming,
                outgoing_pointers=outgoing_pointers,
                incoming_pointers=incoming_pointers,
                status=status,
                signal=signal,
                signal_mr=signal_mr,
                clock_rate_hz=_CudaAllocation.clock_rate_hz(),
            )
        except Exception:
            if signal_mr is not None:
                try:
                    signal_mr.close()
                except Exception:
                    pass
            for allocation in reversed(allocations):
                try:
                    allocation.close()
                except Exception:
                    pass
            for handle in reversed(device_outgoing + device_incoming):
                try:
                    handle.close()
                except Exception:
                    pass
            raise

    def _close_lanes(self) -> None:
        if self._gpunetio is not None:
            try:
                self._gpunetio.close()
            except Exception:
                pass
            self._gpunetio = None
        for lane in reversed(self._outgoing + self._incoming):
            try:
                lane.close()
            except Exception:
                pass
        self._outgoing = []
        self._incoming = []
        self._next_buffers = None

    def reconfigure(self, opts: Any) -> Work:
        """Consume nightly ``ReconfigureOptions`` and rebuild fresh QPs."""

        work = self._reconfigure(
            int(opts.uuid),
            opts.handles,
            timeout=_timeout_value(getattr(opts, "timeout", None)),
            hints=getattr(opts, "hints", None),
        )
        work.wait()
        return Work(None)

    def _reconfigure(
        self,
        uuid: int,
        handles: Iterable[str],
        timeout: Optional[Any] = None,
        hints: Optional[Mapping[str, str]] = None,
    ) -> Work:
        """Build or rebuild the communicator with a c10d-like contract.

        ``uuid`` must be fresh for this process group.  A list or tuple assigns
        ranks by position.  Any other iterable is sorted by ``stable_rank``.
        Every member calls this method concurrently with identical membership.
        The completed :class:`Work` returned by this method has ``rank`` as its
        result.

        Reconfiguration recovers the communicator after a failed or timed-out
        collective; the interrupted tensor is not rolled back and must not be
        consumed.
        """

        del hints
        if self._closed:
            raise ReconfigureError("process group is closed")
        uuid = int(uuid)
        if uuid in self._used_uuids:
            raise ReconfigureError("each reconfigure() requires a fresh uuid")
        ordered_input = isinstance(handles, (list, tuple))
        raw_handles = list(handles)
        if not raw_handles:
            raise ReconfigureError("reconfigure requires at least one handle")
        if len(set(raw_handles)) != len(raw_handles):
            raise ReconfigureError("reconfigure handles must be unique")
        members = [_decode_handle(value) for value in raw_handles]
        if not ordered_input:
            members.sort(key=lambda member: (member.stable_rank, member.raw))
        if sum(member.raw == self._handle for member in members) != 1:
            raise ReconfigureError("local handle is not in the new communicator")
        stable_ranks = [member.stable_rank for member in members]
        if len(set(stable_ranks)) != len(stable_ranks):
            raise ReconfigureError("stable ranks must be unique")
        for member in members:
            if member.qps != self.qps or member.num_sms != self.num_sms:
                raise ReconfigureError("all members must use identical qps and num_sms")
            if member.max_bytes != self.max_bytes:
                raise ReconfigureError(
                    "all members must use identical staging-buffer capacities"
                )

        rank = next(i for i, member in enumerate(members) if member.raw == self._handle)
        size = len(members)
        deadline = time.monotonic() + _seconds(timeout, self.timeout)
        self._close_lanes()
        self._failed = True
        self._rank = -1
        self._size = 0
        self.uuid = None
        if size == 1:
            self._rank = 0
            self._size = 1
            self.uuid = uuid
            self._failed = False
            self._used_uuids.add(uuid)
            return Work(0)

        outgoing, incoming = self._create_lanes()
        next_member = members[(rank + 1) % size]
        prev_member = members[(rank - 1) % size]
        request = {
            "uuid": uuid,
            "handle": self._handle,
            "outgoing": self._lane_infos(outgoing),
            "buffers": self._buffer_info(),
        }
        next_socket = None
        prev_socket = None
        gpunetio_state = None
        try:
            next_socket = _connect_until(next_member.host, next_member.port, deadline)
            next_socket.settimeout(max(0.001, deadline - time.monotonic()))
            _send_frame(next_socket, request)
            prev_socket, prev_request = _accept_member(
                self._listener, prev_member.raw, uuid, deadline
            )
            response = {
                "uuid": uuid,
                "handle": self._handle,
                "incoming": self._lane_infos(incoming),
                "buffers": self._buffer_info(),
            }
            _send_frame(prev_socket, response)
            next_response = _recv_frame(next_socket)
            if (
                next_response.get("uuid") != uuid
                or next_response.get("handle") != next_member.raw
            ):
                raise ReconfigureError("next peer returned mismatched reconfigure data")

            next_infos = [
                _qp_info_from_dict(value) for value in next_response["incoming"]
            ]
            prev_infos = [
                _qp_info_from_dict(value) for value in prev_request["outgoing"]
            ]
            if len(next_infos) != self.qps or len(prev_infos) != self.qps:
                raise ReconfigureError("peer returned the wrong number of QPs")
            retry_count = 3
            rts = {
                "timeout": _ibv_timeout_exponent(self.timeout, retry_count),
                "retry_cnt": retry_count,
                # Ranks may JIT their first CuTe kernel at different speeds.
                # Keep retrying RNR until the receiver posts; the collective's
                # monotonic deadline still terminates a failed operation and
                # reconfigure() destroys the pending QP.
                "rnr_retry": 7,
            }
            access = AccessFlags.LOCAL_WRITE | AccessFlags.REMOTE_WRITE
            for lane, remote in zip(outgoing, next_infos):
                connect_rc(
                    lane.qp,
                    remote,
                    port=self.port,
                    sgid_index=self.gid_index,
                    access=access,
                    **rts,
                )
            for lane, remote in zip(incoming, prev_infos):
                connect_rc(
                    lane.qp,
                    remote,
                    port=self.port,
                    sgid_index=self.gid_index,
                    access=access,
                    **rts,
                )
            buffers = next_response["buffers"]
            next_buffers = {
                key: int(buffers[key])
                for key in ("work_addr", "work_rkey", "scratch_addr", "scratch_rkey")
            }
            gpunetio_state = self._export_gpunetio(outgoing, incoming)
        except Exception:
            if gpunetio_state is not None:
                try:
                    gpunetio_state.close()
                except Exception:
                    pass
            for lane in reversed(outgoing + incoming):
                try:
                    lane.close()
                except Exception:
                    pass
            raise
        finally:
            if next_socket is not None:
                next_socket.close()
            if prev_socket is not None:
                prev_socket.close()

        self._outgoing = outgoing
        self._incoming = incoming
        self._next_buffers = next_buffers
        self._gpunetio = gpunetio_state
        self._rank = rank
        self._size = size
        self.uuid = uuid
        self._failed = False
        self._used_uuids.add(uuid)
        return Work(rank)

    def allreduce(self, tensors: Sequence[torch.Tensor], opts: Any) -> Work:
        """Run an in-place SUM over the one tensor accepted by c10d."""

        if len(tensors) != 1:
            raise ValueError("ibverbs allreduce accepts exactly one tensor")
        reduce_op = getattr(opts, "reduceOp", dist.ReduceOp.SUM)
        if reduce_op != dist.ReduceOp.SUM:
            raise NotImplementedError("ibverbs allreduce only supports SUM")
        self._allreduce_tensor(
            tensors[0], timeout=_timeout_value(getattr(opts, "timeout", None))
        )
        return Work(list(tensors))

    def _allreduce_tensor(
        self, tensor: Any, op: str = "sum", timeout: Optional[Any] = None
    ) -> Any:
        """All-reduce ``tensor`` in place and return it.

        The operation uses a rank-ordered Ring/Simple schedule and exactly
        ``num_sms`` CuTe CTAs. GPUNetIO posts and polls one QP per active CTA
        from a single persistent kernel. ``timeout`` covers CUDA and verbs
        completion, and the completion loops enforce it directly on the GPU.

        A timeout or verbs error puts the group in a failed state.  Exchange a
        surviving set of handles and call :meth:`reconfigure` with a fresh UUID
        before issuing another collective.  The partially modified ``tensor``
        from a failed call is undefined.
        """

        if str(op).lower() != "sum":
            raise NotImplementedError("only SUM all-reduce is implemented")
        if self._closed:
            raise AllReduceError("process group is closed")
        if self.uuid is None or self._size <= 0:
            raise AllReduceError("call reconfigure() before allreduce()")
        if self._failed:
            raise AllReduceError("process group failed; reconfigure it before reuse")
        nbytes = _tensor_nbytes(tensor)
        if nbytes <= 0:
            raise ValueError("cannot all-reduce an empty tensor")
        if nbytes > self.max_bytes:
            raise ValueError("tensor exceeds the configured staging-buffer capacity")
        dtype_key, itemsize = _dtype_key(tensor)
        if nbytes % itemsize:
            raise ValueError("tensor byte length is not a whole number of elements")
        count = nbytes // itemsize
        if count > 0x7FFFFFFF:
            raise ValueError("all-reduce supports at most 2**31 - 1 elements")
        tensor_ptr = int(tensor.data_ptr())
        if tensor_ptr % 16:
            raise ValueError("all-reduce tensor must be at least 16-byte aligned")
        deadline = time.monotonic() + _seconds(timeout, self.timeout)
        work_ptr = int(self.work_buffer.data_ptr())
        scratch_ptr = int(self.scratch_buffer.data_ptr())

        try:
            if tensor_ptr != work_ptr:
                self._kernels.copy(tensor_ptr, work_ptr, count, dtype_key)
                _ibcuda.synchronize()
                _check_deadline(deadline)
            if self._size == 1:
                if tensor_ptr != work_ptr:
                    self._kernels.copy(work_ptr, tensor_ptr, count, dtype_key)
                    _ibcuda.synchronize()
                    _check_deadline(deadline)
                return tensor

            gpunetio = self._gpunetio
            next_buffers = self._next_buffers
            if gpunetio is None or next_buffers is None:
                raise AllReduceError("GPUNetIO communicator is not exported")
            _check_deadline(deadline)
            gpunetio.status.zero()
            timeout_cycles = max(
                1,
                int((deadline - time.monotonic()) * gpunetio.clock_rate_hz),
            )
            self._gpunetio_kernels.allreduce(
                outgoing_qps=gpunetio.outgoing_pointers.ptr,
                incoming_qps=gpunetio.incoming_pointers.ptr,
                status=gpunetio.status.ptr,
                work=work_ptr,
                scratch=scratch_ptr,
                count=count,
                dtype_key=dtype_key,
                itemsize=itemsize,
                rank=self._rank,
                world=self._size,
                remote_buffers=next_buffers,
                work_lkey=self._work_mr.lkey,
                signal_addr=gpunetio.signal_mr.addr,
                signal_lkey=gpunetio.signal_mr.lkey,
                timeout_cycles=timeout_cycles,
            )
            _ibcuda.synchronize()
            statuses = gpunetio.status.read_i32(self.num_sms)
            failed = next((value for value in statuses if value != 0), 0)
            if failed == -110:
                raise AllReduceTimeoutError(
                    "timed out in the GPUNetIO all-reduce kernel"
                )
            if failed:
                raise AllReduceError(
                    "GPUNetIO all-reduce failed with DOCA status %d" % failed
                )
            _check_deadline(deadline)
            if tensor_ptr != work_ptr:
                self._kernels.copy(work_ptr, tensor_ptr, count, dtype_key)
                _ibcuda.synchronize()
                _check_deadline(deadline)
            return tensor
        except Exception:
            self._failed = True
            raise

    def abort(self) -> None:
        """Revoke the current communicator while retaining reconfigure state."""

        if self._closed:
            return
        self._close_lanes()
        self._failed = True
        self._rank = -1
        self._size = 0
        self.uuid = None

    def getRank(self) -> int:
        """Return the rank assigned by the latest reconfiguration."""

        return self._rank

    def getSize(self) -> int:
        """Return the world size assigned by the latest reconfiguration."""

        return self._size

    def getBackendName(self) -> str:
        """Return the registered c10d backend name."""

        return self.backend_name

    def get_backend(self, device: Any) -> "ProcessGroup":
        """Expose this backend through nightly's experimental accessor."""

        del device
        return self

    def set_timeout(self, timeout: timedelta) -> None:
        """Set the default timeout used by future collectives."""

        self.timeout = timeout.total_seconds()

    def shutdown(self) -> None:
        """Release GPUNetIO, verbs, retained CUDA, and HCA resources."""

        self.close()

    def close(self) -> None:
        """Close QPs, deregister staging buffers, and close the listener."""

        if self._closed:
            return
        self._close_lanes()
        self._listener.close()
        self._scratch_mr.close()
        self._work_mr.close()
        self.pd.close()
        self.context.close()
        self._closed = True

    def __enter__(self) -> "ProcessGroup":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


def _create_backend(dist_options: Any, backend_options: Any) -> ProcessGroup:
    if not isinstance(backend_options, ProcessGroupOptions):
        raise TypeError("ibverbs backend requires ProcessGroupOptions")
    stable_rank = backend_options.stable_rank
    global_ranks = list(getattr(dist_options, "global_ranks_in_group", []))
    if stable_rank is None and global_ranks:
        stable_rank = global_ranks[dist_options.group_rank]
        backend_options = replace(backend_options, stable_rank=stable_rank)
    return ProcessGroup.create(
        dist_options.store,
        dist_options.group_rank,
        dist_options.group_size,
        backend_options,
        dist_options.timeout,
        initialize=not bool(dist_options.enable_reconfigure),
        group_id=str(dist_options.group_id),
    )


def register_backend(name: str = "ibverbs") -> str:
    """Register the nightly GPUNetIO PyProcessGroup as a CUDA-only backend."""

    _require_torch_nightly()
    normalized = name.lower()
    plugin = dist.Backend._plugins.get(normalized.upper())
    if plugin is None:
        dist.Backend.register_backend(
            normalized,
            _create_backend,
            extended_api=True,
            devices=["cuda"],
        )
    elif plugin.creator_fn is not _create_backend:
        raise RuntimeError("c10d backend %r is already registered" % normalized)
    return normalized


__all__ = [
    "AllReduceError",
    "AllReduceTimeoutError",
    "ProcessGroup",
    "ProcessGroupOptions",
    "ReconfigureError",
    "Work",
    "register_backend",
]
