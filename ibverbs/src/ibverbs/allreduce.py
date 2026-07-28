"""Optional fault-tolerant CUDA all-reduce over RC ibverbs queue pairs.

This module deliberately is not imported by :mod:`ibverbs`.  It adds no
dependency to the base package: importing it needs only the standard library,
and CuTe DSL is loaded lazily when :meth:`ProcessGroup.allreduce` is first
called.  The transport uses GPUDirect RDMA only; it never uses CUDA IPC,
NVLink, or NCCL.

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
import select
import socket
import struct
import time
import uuid as uuid_module
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import _ibverbs as _ib
from . import cuda as _ibcuda
from .enums import AccessFlags, SendFlags, WROpcode
from .helpers import QPInfo, connect_rc, local_qp_info

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


class AllReduceError(RuntimeError):
    """Base class for communicator and collective failures."""


class AllReduceTimeoutError(TimeoutError, AllReduceError):
    """An all-reduce or reconfiguration exceeded its configured deadline."""


class ReconfigureError(AllReduceError):
    """A membership handle or ring reconstruction was invalid."""


class Work:
    """A c10d-like completed work handle returned by ``reconfigure``.

    Reconfiguration is performed synchronously so that verbs resources cannot
    become visible half-connected.  The returned object still offers the small
    ``Work`` surface useful to callers shared with c10d code.
    """

    def __init__(self, result: Any = None, exception: Optional[BaseException] = None):
        self._result = result
        self._exception = exception

    def is_completed(self) -> bool:
        """Return ``True``; reconfiguration finishes before this is returned."""

        return True

    def is_success(self) -> bool:
        """Return whether the synchronous operation completed successfully."""

        return self._exception is None

    def exception(self) -> Optional[BaseException]:
        """Return the captured exception, if any."""

        return self._exception

    def wait(self, timeout: Optional[Any] = None) -> Any:
        """Return the result or re-raise the operation failure.

        ``timeout`` is accepted for API compatibility.  The work is already
        complete, so it never blocks.
        """

        del timeout
        if self._exception is not None:
            raise self._exception
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
    transport: str


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
            "transport",
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
        or member.transport not in {"gpunetio", "host"}
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


def _poll_one(cq: Any, deadline: float, what: str) -> Any:
    while True:
        completions = cq.poll(1)
        if completions:
            completion = completions[0]
            completion.raise_for_status()
            return completion
        if time.monotonic() >= deadline:
            raise AllReduceTimeoutError("timed out waiting for %s" % what)


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


def _nccl_segments(
    count: int,
    itemsize: int,
    world_size: int,
    chunk_index: int,
    num_sms: int,
) -> List[Tuple[int, int]]:
    """Return byte ranges for one NCCL Ring/Simple rank chunk."""

    if count <= 0:
        return []
    cell, cells_per_channel, active = _nccl_channel_layout(count, itemsize, num_sms)
    base_chunk = max(1, _NCCL_SIMPLE_CHUNK_BYTES // itemsize)
    alignment = max(1, 16 // itemsize)
    ranges = []
    owner = chunk_index % world_size
    for channel in range(active):
        channel_start = channel * cells_per_channel * cell
        channel_count = min(count - channel_start, cells_per_channel * cell)
        if channel_count <= 0:
            continue
        elem_offset = 0
        chunk = base_chunk
        while elem_offset < channel_count:
            remaining = channel_count - elem_offset
            if remaining < world_size * chunk:
                chunk = (remaining + world_size - 1) // world_size
                chunk = ((chunk + alignment - 1) // alignment) * alignment
            local = owner * chunk
            length = min(chunk, max(0, remaining - local))
            if length:
                start = channel_start + elem_offset + local
                ranges.append((start * itemsize, length * itemsize))
            elem_offset += world_size * chunk
    return ranges


def _stripe_segments(
    ranges: Sequence[Tuple[int, int]], lanes: int, alignment: int
) -> Dict[int, List[Tuple[int, int]]]:
    """Split ranges over QPs while preserving non-overlapping byte coverage."""

    striped = {lane: [] for lane in range(lanes)}
    if len(ranges) >= lanes:
        for index, piece in enumerate(ranges):
            striped[index % lanes].append(piece)
        return striped
    if not ranges:
        return {}

    cursor = 0
    lanes_per_range, extra_lanes = divmod(lanes, len(ranges))
    for range_index, (offset, length) in enumerate(ranges):
        if length % alignment:
            raise ValueError("range length must be a multiple of alignment")
        units = length // alignment
        desired = lanes_per_range + (range_index < extra_lanes)
        piece_count = min(desired, units)
        position = offset
        base_units, extra = divmod(units, piece_count)
        for piece_index in range(piece_count):
            piece = (base_units + (piece_index < extra)) * alignment
            lane = cursor % lanes
            striped[lane].append((position, piece))
            cursor += 1
            position += piece
    return {lane: pieces for lane, pieces in striped.items() if pieces}


class _CuteKernels:
    """Lazily JIT-compiled CuTe copy and NCCL-ordered reduction kernels."""

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
        self._compiled: Dict[str, Tuple[Any, Any, Any, Any, Any]] = {}

    def _get(self, dtype_key: str, pointer: int) -> Tuple[Any, Any, Any, Any, Any]:
        cached = self._compiled.get(dtype_key)
        if cached is not None:
            return cached
        try:
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import make_ptr
        except ImportError as exc:
            raise RuntimeError(
                "CuTe all-reduce requires Python 3.10+ and the optional "
                "nvidia-cutlass-dsl package; install with "
                "pip install './ibverbs[allreduce]'"
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

        @cute.kernel
        def reduce_kernel(
            work: cute.Tensor,
            scratch: cute.Tensor,
            count: cutlass.Int32,
            world: cutlass.Int32,
            owner: cutlass.Int32,
            cell_elements: cutlass.Int32,
            cells_per_channel: cutlass.Int32,
            active_channels: cutlass.Int32,
            base_chunk: cutlass.Int32,
            align_elements: cutlass.Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            channel, _, _ = cute.arch.block_idx()
            if channel < active_channels:
                channel_start = (
                    cutlass.Int32(channel) * cells_per_channel * cell_elements
                )
                channel_count = min(
                    count - channel_start, cells_per_channel * cell_elements
                )
                loop_count = cutlass.Int32(world) * base_chunk
                for elem_offset in cutlass.range(0, channel_count, loop_count):
                    remaining = channel_count - elem_offset
                    chunk = base_chunk
                    if remaining < loop_count:
                        chunk = (remaining + cutlass.Int32(world) - 1) // cutlass.Int32(
                            world
                        )
                        chunk = (
                            (chunk + align_elements - 1) // align_elements
                        ) * align_elements
                    local = cutlass.Int32(owner) * chunk
                    length = min(chunk, max(cutlass.Int32(0), remaining - local))
                    start = channel_start + elem_offset + local
                    for i in cutlass.range(
                        cutlass.Int32(tid), length, cutlass.Int32(256)
                    ):
                        index = start + i
                        if index < count:
                            work[index] = work[index] + scratch[index]

        @cute.jit
        def reduce_host(
            work_ptr: cute.Pointer,
            scratch_ptr: cute.Pointer,
            count: cutlass.Int32,
            world: cutlass.Int32,
            owner: cutlass.Int32,
            cell_elements: cutlass.Int32,
            cells_per_channel: cutlass.Int32,
            active_channels: cutlass.Int32,
            base_chunk: cutlass.Int32,
            align_elements: cutlass.Int32,
        ):
            work = cute.make_tensor(work_ptr, cute.make_layout((count,)))
            scratch = cute.make_tensor(scratch_ptr, cute.make_layout((count,)))
            reduce_kernel(
                work,
                scratch,
                count,
                world,
                owner,
                cell_elements,
                cells_per_channel,
                active_channels,
                base_chunk,
                align_elements,
            ).launch(grid=[sms, 1, 1], block=[256, 1, 1])

        ptr = make_ptr(dtype, pointer, cute.AddressSpace.gmem, assumed_align=16)
        copy = cute.compile(copy_host, ptr, ptr, cutlass.Int32(1))
        reduce = cute.compile(
            reduce_host,
            ptr,
            ptr,
            cutlass.Int32(1),
            cutlass.Int32(2),
            cutlass.Int32(0),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
        )
        cached = (copy, reduce, make_ptr, dtype, cutlass)
        self._compiled[dtype_key] = cached
        return cached

    def copy(self, source: int, destination: int, count: int, dtype_key: str) -> None:
        """Copy ``count`` elements with a fixed ``num_sms`` CTA launch."""

        copy, _, make_ptr, dtype, cutlass = self._get(dtype_key, source)
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

    def reduce(
        self,
        work: int,
        scratch: int,
        count: int,
        dtype_key: str,
        itemsize: int,
        world: int,
        owner: int,
    ) -> None:
        """Reduce the selected NCCL ring chunks from scratch into work."""

        _, reduce, make_ptr, dtype, cutlass = self._get(dtype_key, work)
        ptr_work = make_ptr(
            dtype, work, self._address_space(make_ptr), assumed_align=16
        )
        ptr_scratch = make_ptr(
            dtype, scratch, self._address_space(make_ptr), assumed_align=16
        )
        cell, cells_per_channel, active = _nccl_channel_layout(
            count, itemsize, self.num_sms
        )
        reduce(
            ptr_work,
            ptr_scratch,
            cutlass.Int32(count),
            cutlass.Int32(world),
            cutlass.Int32(owner),
            cutlass.Int32(cell),
            cutlass.Int32(cells_per_channel),
            cutlass.Int32(active),
            cutlass.Int32(max(1, _NCCL_SIMPLE_CHUNK_BYTES // itemsize)),
            cutlass.Int32(max(1, 16 // itemsize)),
        )


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
        dtype = getattr(cutlass, _CuteKernels._CUTLASS_TYPES[dtype_key])
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


class ProcessGroup:
    """A reconfigurable, rank-ordered GPUDirect RDMA all-reduce group.

    Args:
        context: Open :class:`ibverbs.Context` for the GPU-local HCA.
        pd: Protection domain allocated from ``context``.
        work_buffer: Contiguous VMM-backed CUDA byte tensor used internally.
        scratch_buffer: A distinct buffer with at least the same capacity.
        stable_rank: Stable application rank encoded in reconfigure handles.
        gid_index: Local GID index used to connect RC QPs.
        port: Active HCA port number.
        advertise_host: Address peers can use for the reconfigure rendezvous.
        bind_host: Local address for the rendezvous listener.
        rendezvous_port: Listener port, or zero for an ephemeral port.
        qps: RC queue pairs used in parallel for each ring direction. By
            default GPUNetIO uses one QP per channel and the host path uses 4.
        num_sms: CuTe CTA count.  The benchmark forces NCCL to this same count.
        timeout: Default all-reduce and reconfiguration deadline.
        queue_depth: Maximum posted writes per QP in one ring round.
        transport: ``"gpunetio"`` for a persistent GPU-posted ring or
            ``"host"`` for the portable host-posted fallback.
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

    supports_reconfigure = True

    def __init__(
        self,
        context: Any,
        pd: Any,
        work_buffer: Any,
        scratch_buffer: Any,
        *,
        stable_rank: int,
        gid_index: int,
        port: int = 1,
        advertise_host: Optional[str] = None,
        bind_host: str = "0.0.0.0",
        rendezvous_port: int = 0,
        qps: Optional[int] = None,
        num_sms: int = 32,
        timeout: Any = 30.0,
        queue_depth: int = 256,
        transport: str = "gpunetio",
        gpu: Any = None,
        gpunetio_arch: str = "sm_90",
        gpunetio_bitcode: Any = None,
        nic_handler: str = "gpu",
    ):
        transport = str(transport).lower()
        if transport not in {"gpunetio", "host"}:
            raise ValueError("transport must be 'gpunetio' or 'host'")
        num_sms = int(num_sms)
        if qps is None:
            qps = num_sms if transport == "gpunetio" else 4
        if int(stable_rank) < 0:
            raise ValueError("stable_rank must be non-negative")
        if int(qps) <= 0 or num_sms <= 0 or int(queue_depth) <= 0:
            raise ValueError("qps, num_sms, and queue_depth must be positive")
        if transport == "gpunetio" and int(qps) < num_sms:
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
        self.transport = transport
        self.gpu = gpu
        self.gpunetio_arch = str(gpunetio_arch)
        self.gpunetio_bitcode = gpunetio_bitcode
        self.nic_handler = nic_handler
        self.max_bytes = work_bytes
        self.rank = -1
        self.size = 0
        self.uuid: Optional[int] = None
        self._sequence = 0
        self._failed = False
        self._closed = False
        self._outgoing: List[_Lane] = []
        self._incoming: List[_Lane] = []
        self._next_buffers: Optional[Dict[str, int]] = None
        self._gpunetio: Optional[_GpuNetIOState] = None
        self._used_uuids = set()
        self._kernels = _CuteKernels(self.num_sms)
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
                "transport": self.transport,
            }
        )

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
                            _ib.QPInitAttr(
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

        from .gpunetio import DeviceQP

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

    def reconfigure(
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
            if member.transport != self.transport:
                raise ReconfigureError("all members must use the same transport")
            if member.max_bytes != self.max_bytes:
                raise ReconfigureError(
                    "all members must use identical staging-buffer capacities"
                )

        rank = next(i for i, member in enumerate(members) if member.raw == self._handle)
        size = len(members)
        deadline = time.monotonic() + _seconds(timeout, self.timeout)
        self._close_lanes()
        self._failed = True
        self.rank = -1
        self.size = 0
        self.uuid = None
        if size == 1:
            self.rank = 0
            self.size = 1
            self.uuid = uuid
            self._failed = False
            self._sequence = 0
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
            if self.transport == "gpunetio":
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
        self.rank = rank
        self.size = size
        self.uuid = uuid
        self._failed = False
        self._sequence = 0
        self._used_uuids.add(uuid)
        return Work(rank)

    def _notification(self, phase: int, round_index: int) -> int:
        if round_index >= 0x4000:
            raise AllReduceError("communicator is too large for notification encoding")
        return (
            ((self._sequence & 0xFFFF) << 16)
            | ((phase & 0x3) << 14)
            | (round_index & 0x3FFF)
        )

    def _round(
        self,
        source_ranges: Sequence[Tuple[int, int]],
        receive_ranges: Sequence[Tuple[int, int]],
        destination: str,
        notification: int,
        deadline: float,
        alignment: int,
    ) -> None:
        assert self._next_buffers is not None
        send_stripes = _stripe_segments(source_ranges, self.qps, alignment)
        recv_stripes = _stripe_segments(receive_ranges, self.qps, alignment)
        if not send_stripes:
            send_stripes = {0: []}
        if not recv_stripes:
            recv_stripes = {0: []}

        for lane_index in recv_stripes:
            self._incoming[lane_index].qp.post_recv(
                _ib.RecvWR(wr_id=notification, sg_list=[])
            )

        remote_addr = self._next_buffers[destination + "_addr"]
        remote_rkey = self._next_buffers[destination + "_rkey"]
        for lane_index, pieces in send_stripes.items():
            writes = []
            if len(pieces) > self.queue_depth:
                raise AllReduceError(
                    "ring round exceeds queue_depth; increase queue_depth or reduce max_bytes"
                )
            for index, (offset, length) in enumerate(pieces):
                last = index == len(pieces) - 1
                writes.append(
                    _ib.SendWR(
                        wr_id=notification,
                        sg_list=[self._work_mr.sge(length, offset=offset)],
                        opcode=(
                            WROpcode.RDMA_WRITE_WITH_IMM
                            if last
                            else WROpcode.RDMA_WRITE
                        ),
                        send_flags=SendFlags.SIGNALED if last else 0,
                        remote_addr=remote_addr + offset,
                        rkey=remote_rkey,
                        imm_data=notification if last else 0,
                    )
                )
            if not writes:
                writes.append(
                    _ib.SendWR(
                        wr_id=notification,
                        sg_list=[],
                        opcode=WROpcode.RDMA_WRITE_WITH_IMM,
                        send_flags=SendFlags.SIGNALED,
                        remote_addr=remote_addr,
                        rkey=remote_rkey,
                        imm_data=notification,
                    )
                )
            self._outgoing[lane_index].qp.post_send(writes)

        for lane_index in send_stripes:
            completion = _poll_one(
                self._outgoing[lane_index].cq, deadline, "RDMA send completion"
            )
            if completion.wr_id != notification:
                raise AllReduceError("stale send completion in all-reduce")
        for lane_index in recv_stripes:
            completion = _poll_one(
                self._incoming[lane_index].cq, deadline, "RDMA receive notification"
            )
            if completion.imm_data != notification:
                raise AllReduceError("stale receive notification in all-reduce")

    def allreduce(
        self, tensor: Any, op: str = "sum", timeout: Optional[Any] = None
    ) -> Any:
        """All-reduce ``tensor`` in place and return it.

        The operation uses a rank-ordered Ring/Simple schedule and exactly
        ``num_sms`` CuTe CTAs. GPUNetIO posts and polls one QP per active CTA
        from a single persistent kernel; the fallback stripes host-posted work
        over ``qps`` QPs. ``timeout`` covers CUDA and verbs completion, and the
        GPUNetIO completion loops enforce it directly on the GPU.

        A timeout or verbs error puts the group in a failed state.  Exchange a
        surviving set of handles and call :meth:`reconfigure` with a fresh UUID
        before issuing another collective.  The partially modified ``tensor``
        from a failed call is undefined.
        """

        if str(op).lower() != "sum":
            raise NotImplementedError("only SUM all-reduce is implemented")
        if self._closed:
            raise AllReduceError("process group is closed")
        if self.uuid is None or self.size <= 0:
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
            if self.size == 1:
                if tensor_ptr != work_ptr:
                    self._kernels.copy(work_ptr, tensor_ptr, count, dtype_key)
                    _ibcuda.synchronize()
                    _check_deadline(deadline)
                self._sequence += 1
                return tensor

            if self.transport == "gpunetio":
                if self._gpunetio is None or self._next_buffers is None:
                    raise AllReduceError("GPUNetIO communicator is not exported")
                _check_deadline(deadline)
                self._gpunetio.status.zero()
                timeout_cycles = max(
                    1,
                    int((deadline - time.monotonic()) * self._gpunetio.clock_rate_hz),
                )
                self._gpunetio_kernels.allreduce(
                    outgoing_qps=self._gpunetio.outgoing_pointers.ptr,
                    incoming_qps=self._gpunetio.incoming_pointers.ptr,
                    status=self._gpunetio.status.ptr,
                    work=work_ptr,
                    scratch=scratch_ptr,
                    count=count,
                    dtype_key=dtype_key,
                    itemsize=itemsize,
                    rank=self.rank,
                    world=self.size,
                    remote_buffers=self._next_buffers,
                    work_lkey=self._work_mr.lkey,
                    signal_addr=self._gpunetio.signal_mr.addr,
                    signal_lkey=self._gpunetio.signal_mr.lkey,
                    timeout_cycles=timeout_cycles,
                )
                _ibcuda.synchronize()
                statuses = self._gpunetio.status.read_i32(self.num_sms)
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
                self._sequence += 1
                return tensor

            # Reduce-scatter.  Round k sends rank-k and receives rank-k-1;
            # this is the exact chunk sequence in NCCL's Ring/Simple kernel.
            for k in range(1, self.size):
                send_owner = (self.rank - k) % self.size
                recv_owner = (self.rank - k - 1) % self.size
                self._round(
                    _nccl_segments(
                        count, itemsize, self.size, send_owner, self.num_sms
                    ),
                    _nccl_segments(
                        count, itemsize, self.size, recv_owner, self.num_sms
                    ),
                    "scratch",
                    self._notification(0, k - 1),
                    deadline,
                    itemsize,
                )
                _ibcuda.flush_gpudirect_writes()
                self._kernels.reduce(
                    work_ptr,
                    scratch_ptr,
                    count,
                    dtype_key,
                    itemsize,
                    self.size,
                    recv_owner,
                )
                _ibcuda.synchronize()
                _check_deadline(deadline)

            # All-gather.  Inbound writes land directly in the final work
            # buffer; the flush orders them before the next QP reads it.
            for k in range(0, self.size - 1):
                send_owner = (self.rank - k) % self.size
                recv_owner = (self.rank - k - 1) % self.size
                self._round(
                    _nccl_segments(
                        count, itemsize, self.size, send_owner, self.num_sms
                    ),
                    _nccl_segments(
                        count, itemsize, self.size, recv_owner, self.num_sms
                    ),
                    "work",
                    self._notification(1, k),
                    deadline,
                    itemsize,
                )
                _ibcuda.flush_gpudirect_writes()
                _ibcuda.synchronize()
                _check_deadline(deadline)

            if tensor_ptr != work_ptr:
                self._kernels.copy(work_ptr, tensor_ptr, count, dtype_key)
                _ibcuda.synchronize()
                _check_deadline(deadline)
            self._sequence += 1
            return tensor
        except Exception:
            self._failed = True
            raise

    def close(self) -> None:
        """Close QPs, deregister staging buffers, and close the listener."""

        if self._closed:
            return
        self._close_lanes()
        self._listener.close()
        self._scratch_mr.close()
        self._work_mr.close()
        self._closed = True

    def __enter__(self) -> "ProcessGroup":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


__all__ = [
    "AllReduceError",
    "AllReduceTimeoutError",
    "ProcessGroup",
    "ReconfigureError",
    "Work",
]
