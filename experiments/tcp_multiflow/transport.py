"""Striped multi-flow TCP transport for contiguous CPU tensor buffers."""

from __future__ import annotations

import ipaddress
import mmap
import multiprocessing
import os
import secrets
import socket
import struct
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from experiments.tcp_multiflow import benchmark

_MAGIC = b"TENSOR01"
_DONE = b"TEND0001"
_HEADER = struct.Struct("!8s16sIQQQ")
_SHARED_TENSOR_IDS = set()


def _forget_shared_tensor(identity: int) -> None:
    _SHARED_TENSOR_IDS.discard(identity)


@dataclass(frozen=True)
class NetworkEndpoint:
    """A local interface and address used by one side of a transfer."""

    interface: str
    address: str


@dataclass(frozen=True)
class FlowTransfer:
    """Transfer statistics for one striped TCP flow."""

    flow: int
    port: int
    offset: int
    nbytes: int
    seconds: float
    retransmits: int

    @property
    def gbps(self) -> float:
        """Return application payload throughput for this flow."""

        return self.nbytes * 8 / self.seconds / 1e9


@dataclass(frozen=True)
class SendResult:
    """Aggregate result returned by :meth:`TensorSender.send`."""

    nbytes: int
    seconds: float
    flows: Tuple[FlowTransfer, ...]

    @property
    def gbps(self) -> float:
        """Return aggregate application payload throughput."""

        return self.nbytes * 8 / self.seconds / 1e9

    @property
    def retransmits(self) -> int:
        """Return the aggregate Linux TCP retransmit count."""

        return sum(flow.retransmits for flow in self.flows)


@dataclass(frozen=True)
class ReceiveResult:
    """Aggregate result returned by :meth:`TensorReceiver.receive_into`."""

    nbytes: int
    seconds: float
    completion_seconds: float

    @property
    def gbps(self) -> float:
        """Return aggregate network payload throughput."""

        return self.nbytes * 8 / self.seconds / 1e9

    @property
    def completion_gbps(self) -> float:
        """Return payload throughput including final tensor placement."""

        return self.nbytes * 8 / self.completion_seconds / 1e9


@dataclass(frozen=True)
class LocalTransferResult:
    """Application and physical-path result for a same-host tensor transfer."""

    nbytes: int
    seconds: float
    completion_seconds: float
    payload_gbps: float
    completion_gbps: float
    total_seconds: float
    total_gbps: float
    wire_gbps: float
    tx_wire_gbps: float
    rx_wire_gbps: float
    retransmits: int
    path_verified: bool
    flows: Tuple[FlowTransfer, ...]


class SharedTensor:
    """Process-shared CPU storage for the transport's line-rate fast path.

    Use :meth:`numpy` or :meth:`torch` to expose the same storage as a tensor
    without copying. Keep this object alive for as long as any derived tensor is
    in use.
    """

    def __init__(self, nbytes: int) -> None:
        """Allocate ``nbytes`` of anonymous process-shared memory."""

        if nbytes <= 0:
            raise ValueError("shared tensor size must be positive")
        self.nbytes = nbytes
        self._mapping = mmap.mmap(-1, nbytes)
        self._mapping.madvise(mmap.MADV_HUGEPAGE)

    @property
    def buffer(self) -> memoryview:
        """Return a writable byte view of the shared storage."""

        if self._mapping is None:
            raise RuntimeError("shared tensor storage is closed")
        return memoryview(self._mapping)

    def numpy(self, shape, dtype="uint8"):
        """Return a zero-copy NumPy tensor over the shared storage."""

        import numpy

        if self._mapping is None:
            raise RuntimeError("shared tensor storage is closed")
        tensor = numpy.ndarray(shape, dtype=dtype, buffer=self._mapping)
        if tensor.nbytes != self.nbytes:
            raise ValueError("NumPy shape and dtype do not cover the shared storage")
        return tensor

    def torch(self, shape, dtype=None):
        """Return a zero-copy CPU PyTorch tensor over the shared storage."""

        import torch

        if self._mapping is None:
            raise RuntimeError("shared tensor storage is closed")
        selected_dtype = dtype if dtype is not None else torch.uint8
        tensor = torch.frombuffer(self._mapping, dtype=selected_dtype).reshape(shape)
        if tensor.numel() * tensor.element_size() != self.nbytes:
            raise ValueError("PyTorch shape and dtype do not cover the shared storage")
        identity = id(tensor)
        _SHARED_TENSOR_IDS.add(identity)
        weakref.finalize(tensor, _forget_shared_tensor, identity)
        return tensor

    def close(self) -> None:
        """Release the mapping after all derived tensor views are gone."""

        if self._mapping is not None:
            mapping = self._mapping
            mapping.close()
            self._mapping = None

    def __enter__(self):
        """Return this allocation from a context manager."""

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release the mapping when leaving a context manager."""

        self.close()


@dataclass(frozen=True)
class _TensorBuffer:
    view: memoryview
    owner: Any
    shared: bool


def _is_process_shared(tensor: Any) -> bool:
    if id(tensor) in _SHARED_TENSOR_IDS:
        return True
    candidate = tensor
    visited = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        if isinstance(candidate, (SharedTensor, mmap.mmap)):
            return True
        is_shared = getattr(candidate, "is_shared", None)
        if is_shared is not None and bool(is_shared()):
            return True
        base = getattr(candidate, "base", None)
        candidate = base if base is not None else getattr(candidate, "_base", None)
    return False


def _tensor_buffer(tensor: Any, writable: bool) -> _TensorBuffer:
    if isinstance(tensor, SharedTensor):
        return _TensorBuffer(tensor.buffer, tensor, True)
    owner = tensor
    shared = _is_process_shared(tensor)
    try:
        view = memoryview(tensor)
    except TypeError:
        candidate = tensor
        detach = getattr(candidate, "detach", None)
        if detach is not None:
            candidate = detach()
        device = getattr(candidate, "device", None)
        if device is not None and getattr(device, "type", str(device)) != "cpu":
            raise ValueError("TCP tensor transport requires a CPU tensor")
        is_contiguous = getattr(candidate, "is_contiguous", None)
        if is_contiguous is not None and not is_contiguous():
            raise ValueError("TCP tensor transport requires a contiguous tensor")
        is_shared = getattr(candidate, "is_shared", None)
        if is_shared is not None:
            shared = bool(is_shared())
        numpy = getattr(candidate, "numpy", None)
        if numpy is None:
            raise TypeError("tensor must export a contiguous buffer or numpy() view")
        owner = numpy()
        try:
            view = memoryview(owner)
        except TypeError as error:
            raise TypeError("tensor numpy() result does not export a buffer") from error
    if not view.c_contiguous:
        raise ValueError("TCP tensor transport requires a C-contiguous tensor")
    if writable and view.readonly:
        raise ValueError("destination tensor buffer is read-only")
    try:
        byte_view = view.cast("B")
    except TypeError as error:
        raise ValueError("tensor buffer cannot be represented as bytes") from error
    return _TensorBuffer(byte_view, owner, shared)


def _family(address: str):
    return (
        socket.AF_INET6
        if ipaddress.ip_address(address).version == 6
        else socket.AF_INET
    )


def _address(address: str, port: int):
    if ipaddress.ip_address(address).version == 6:
        return (address, port, 0, 0)
    return (address, port)


def _bind_device(sock: socket.socket, interface: str):
    option = getattr(socket, "SO_BINDTODEVICE", 25)
    sock.setsockopt(socket.SOL_SOCKET, option, interface.encode() + b"\0")


def _partitions(nbytes: int, flows: int) -> List[Tuple[int, int]]:
    if flows <= 0:
        raise ValueError("flows must be positive")
    if nbytes < flows:
        raise ValueError("tensor must contain at least one byte per flow")
    return [
        (nbytes * flow // flows, nbytes * (flow + 1) // flows) for flow in range(flows)
    ]


def _set_affinity(cpu: Optional[int]):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    result = bytearray(nbytes)
    view = memoryview(result)
    offset = 0
    while offset < nbytes:
        received = sock.recv_into(view[offset:])
        if not received:
            raise RuntimeError("TCP peer closed during the transfer header")
        offset += received
    return bytes(result)


def _total_retransmits(sock: socket.socket) -> int:
    option = getattr(socket, "TCP_INFO", None)
    if option is None:
        return 0
    info = sock.getsockopt(socket.IPPROTO_TCP, option, 104)
    if len(info) < 104:
        return 0
    return struct.unpack_from("=I", info, 100)[0]


class TensorSender:
    """Send a contiguous CPU tensor over parallel, interface-bound TCP flows.

    The receiver must listen on the same consecutive port range and provide a
    destination tensor with exactly the same number of bytes. Connections are
    opened for each call to :meth:`send`.
    """

    def __init__(
        self,
        local: NetworkEndpoint,
        remote_address: str,
        *,
        flows: int = 40,
        base_port: int = 6201,
        chunk_bytes: int = 1 << 20,
        stripe_bytes: int = 16 << 20,
        cpus: Optional[Sequence[int]] = None,
        timeout: float = 120.0,
        worker_mode: str = "thread",
    ) -> None:
        """Configure a tensor sender without opening network connections."""

        if flows <= 0 or base_port <= 0 or base_port + flows - 1 > 65535:
            raise ValueError("invalid TCP flow count or port range")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if stripe_bytes <= 0:
            raise ValueError("stripe_bytes must be positive")
        if (
            ipaddress.ip_address(local.address).version
            != ipaddress.ip_address(remote_address).version
        ):
            raise ValueError("local and remote addresses use different IP versions")
        if cpus is not None and len(cpus) != flows:
            raise ValueError("provide exactly one sender CPU per flow")
        if worker_mode not in {"thread", "process"}:
            raise ValueError("worker_mode must be thread or process")
        self.local = local
        self.remote_address = remote_address
        self.flows = flows
        self.base_port = base_port
        self.chunk_bytes = chunk_bytes
        self.stripe_bytes = stripe_bytes
        self.cpus = tuple(cpus) if cpus is not None else (None,) * flows
        self.timeout = timeout
        self.worker_mode = worker_mode

    def _send_flow(
        self,
        view: memoryview,
        transfer_id: bytes,
        flow: int,
        offset: int,
        end: int,
        iterations: int,
        ready: threading.Barrier,
        go: threading.Event,
    ) -> FlowTransfer:
        _set_affinity(self.cpus[flow])
        family = _family(self.local.address)
        port = self.base_port + flow
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            _bind_device(sock, self.local.interface)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.timeout)
            sock.bind(_address(self.local.address, 0))
            sock.connect(_address(self.remote_address, port))
            ready.wait(timeout=self.timeout)
            go.wait()
            sock.settimeout(None)
            start_ns = time.perf_counter_ns()
            sock.sendall(
                _HEADER.pack(
                    _MAGIC,
                    transfer_id,
                    flow,
                    len(view),
                    offset,
                    end - offset,
                )
            )
            for _ in range(iterations):
                position = offset
                while position < end:
                    next_position = min(position + self.chunk_bytes, end)
                    sock.sendall(view[position:next_position])
                    position = next_position
            retransmits = _total_retransmits(sock)
            seconds = (time.perf_counter_ns() - start_ns) / 1e9
        return FlowTransfer(
            flow=flow,
            port=port,
            offset=offset,
            nbytes=(end - offset) * iterations,
            seconds=seconds,
            retransmits=retransmits,
        )

    def _send_dynamic_flow(
        self,
        view,
        transfer_id,
        flow,
        next_task,
        task_count,
        ready,
        go,
    ) -> FlowTransfer:
        _set_affinity(self.cpus[flow])
        staging = bytes(view)
        staging_view = memoryview(staging)
        family = _family(self.local.address)
        port = self.base_port + flow
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            _bind_device(sock, self.local.interface)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.timeout)
            sock.bind(_address(self.local.address, 0))
            sock.connect(_address(self.remote_address, port))
            ready.wait(timeout=self.timeout)
            go.wait()
            sock.settimeout(None)
            start_ns = time.perf_counter_ns()
            sent_bytes = 0
            while True:
                with next_task.get_lock():
                    task = next_task.value
                    next_task.value += 1
                if task >= task_count:
                    break
                stripe = task % (
                    (len(view) + self.stripe_bytes - 1) // self.stripe_bytes
                )
                offset = stripe * self.stripe_bytes
                end = min(offset + self.stripe_bytes, len(view))
                sock.sendall(
                    _HEADER.pack(
                        _MAGIC,
                        transfer_id,
                        flow,
                        len(view),
                        offset,
                        end - offset,
                    )
                )
                position = offset
                while position < end:
                    next_position = min(position + self.chunk_bytes, end)
                    sock.sendall(staging_view[position:next_position])
                    position = next_position
                sent_bytes += end - offset
            sock.sendall(_HEADER.pack(_DONE, transfer_id, flow, len(view), 0, 0))
            retransmits = _total_retransmits(sock)
            seconds = max((time.perf_counter_ns() - start_ns) / 1e9, 1e-9)
        return FlowTransfer(
            flow=flow,
            port=port,
            offset=0,
            nbytes=sent_bytes,
            seconds=seconds,
            retransmits=retransmits,
        )

    def _send_process_entry(
        self,
        result_pipe,
        view,
        transfer_id,
        flow,
        next_task,
        task_count,
        ready,
        go,
    ):
        try:
            result_pipe.send(
                (
                    True,
                    self._send_dynamic_flow(
                        view,
                        transfer_id,
                        flow,
                        next_task,
                        task_count,
                        ready,
                        go,
                    ),
                )
            )
        except BaseException as error:
            result_pipe.send((False, "%s: %s" % (type(error).__name__, error)))
        finally:
            result_pipe.close()

    def _send_processes(
        self,
        view: memoryview,
        transfer_id: bytes,
        iterations: int,
    ) -> SendResult:
        context = multiprocessing.get_context("fork")
        ready = context.Barrier(self.flows + 1)
        go = context.Event()
        stripe_count = (len(view) + self.stripe_bytes - 1) // self.stripe_bytes
        task_count = stripe_count * iterations
        next_task = context.Value("Q", 0)
        processes = []
        readers = []
        try:
            for flow in range(self.flows):
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(
                    target=self._send_process_entry,
                    args=(
                        writer,
                        view,
                        transfer_id,
                        flow,
                        next_task,
                        task_count,
                        ready,
                        go,
                    ),
                )
                process.start()
                writer.close()
                processes.append(process)
                readers.append(reader)
            ready.wait(timeout=self.timeout)
            start_ns = time.perf_counter_ns()
            go.set()
            messages = [reader.recv() for reader in readers]
            seconds = (time.perf_counter_ns() - start_ns) / 1e9
            for process in processes:
                process.join(timeout=self.timeout)
            failures = [message for succeeded, message in messages if not succeeded]
            if failures:
                raise RuntimeError("; ".join(failures))
            flow_results = tuple(message for succeeded, message in messages)
            if sum(result.nbytes for result in flow_results) != len(view) * iterations:
                raise RuntimeError("dynamic tensor workers sent an incomplete transfer")
            return SendResult(len(view) * iterations, seconds, flow_results)
        finally:
            for reader in readers:
                reader.close()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()

    def send(self, tensor: Any, *, iterations: int = 1) -> SendResult:
        """Send a tensor one or more times over persistent flow connections."""

        buffer = _tensor_buffer(tensor, writable=False)
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        transfer_id = secrets.token_bytes(16)
        if self.worker_mode == "process":
            return self._send_processes(
                buffer.view,
                transfer_id,
                iterations,
            )
        partitions = _partitions(len(buffer.view), self.flows)
        ready = threading.Barrier(self.flows + 1)
        go = threading.Event()
        with ThreadPoolExecutor(max_workers=self.flows) as executor:
            futures = [
                executor.submit(
                    self._send_flow,
                    buffer.view,
                    transfer_id,
                    flow,
                    offset,
                    end,
                    iterations,
                    ready,
                    go,
                )
                for flow, (offset, end) in enumerate(partitions)
            ]
            ready.wait(timeout=self.timeout)
            start_ns = time.perf_counter_ns()
            go.set()
            flow_results = tuple(future.result() for future in futures)
            seconds = (time.perf_counter_ns() - start_ns) / 1e9
        return SendResult(len(buffer.view) * iterations, seconds, flow_results)


class TensorReceiver:
    """Receive striped TCP flows directly into a preallocated CPU tensor."""

    def __init__(
        self,
        local: NetworkEndpoint,
        *,
        flows: int = 40,
        base_port: int = 6201,
        chunk_bytes: int = 1 << 20,
        stripe_bytes: int = 16 << 20,
        cpus: Optional[Sequence[int]] = None,
        timeout: float = 120.0,
        worker_mode: str = "thread",
        assume_shared: bool = False,
    ) -> None:
        """Configure a tensor receiver without opening listening sockets."""

        if flows <= 0 or base_port <= 0 or base_port + flows - 1 > 65535:
            raise ValueError("invalid TCP flow count or port range")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if stripe_bytes <= 0:
            raise ValueError("stripe_bytes must be positive")
        if cpus is not None and len(cpus) != flows:
            raise ValueError("provide exactly one receiver CPU per flow")
        if worker_mode not in {"thread", "process"}:
            raise ValueError("worker_mode must be thread or process")
        self.local = local
        self.flows = flows
        self.base_port = base_port
        self.chunk_bytes = chunk_bytes
        self.stripe_bytes = stripe_bytes
        self.cpus = tuple(cpus) if cpus is not None else (None,) * flows
        self.timeout = timeout
        self.worker_mode = worker_mode
        self.assume_shared = assume_shared

    def _receive_flow(
        self,
        listener: socket.socket,
        view: memoryview,
        expected_flow: int,
        expected_offset: int,
        expected_end: int,
        iterations: int,
    ) -> Tuple[bytes, int, int, int, int]:
        _set_affinity(self.cpus[expected_flow])
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(None)
            expected = (
                _MAGIC,
                expected_flow,
                len(view),
                expected_offset,
                expected_end - expected_offset,
            )
            header = _HEADER.unpack(_recv_exact(connection, _HEADER.size))
            magic, transfer_id, flow, total, offset, nbytes = header
            if (magic, flow, total, offset, nbytes) != expected:
                raise RuntimeError(
                    "tensor flow %d header does not match the destination"
                    % expected_flow
                )
            for iteration in range(iterations):
                if iteration == 0:
                    start_ns = time.perf_counter_ns()
                position = expected_offset
                while position < expected_end:
                    next_position = min(position + self.chunk_bytes, expected_end)
                    received = connection.recv_into(view[position:next_position])
                    if not received:
                        raise RuntimeError(
                            "TCP peer closed during tensor flow %d" % expected_flow
                        )
                    position += received
            end_ns = time.perf_counter_ns()
        return (
            transfer_id,
            (expected_end - expected_offset) * iterations,
            start_ns,
            end_ns,
            end_ns,
        )

    def _receive_dynamic_flow(
        self,
        listener: socket.socket,
        destination: memoryview,
        expected_flow: int,
        placed,
    ) -> Tuple[bytes, int, int, int, int]:
        """Receive dynamically assigned stripes and place each destination once."""

        _set_affinity(self.cpus[expected_flow])
        staging = bytearray(min(self.stripe_bytes, len(destination)))
        staging_view = memoryview(staging)
        locally_seen = set()
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(None)
            transfer_id = None
            received_bytes = 0
            start_ns = None
            while True:
                header = _HEADER.unpack(_recv_exact(connection, _HEADER.size))
                magic, current_transfer_id, flow, total, offset, nbytes = header
                if (
                    flow != expected_flow
                    or total != len(destination)
                    or (transfer_id is not None and current_transfer_id != transfer_id)
                ):
                    raise RuntimeError(
                        "tensor flow %d header does not match the destination"
                        % expected_flow
                    )
                transfer_id = current_transfer_id
                if magic == _DONE:
                    if offset != 0 or nbytes != 0:
                        raise RuntimeError("invalid tensor flow terminator")
                    break
                expected_nbytes = min(self.stripe_bytes, total - offset)
                if (
                    magic != _MAGIC
                    or offset % self.stripe_bytes
                    or nbytes != expected_nbytes
                    or nbytes <= 0
                ):
                    raise RuntimeError("invalid dynamic tensor stripe")
                if start_ns is None:
                    start_ns = time.perf_counter_ns()
                position = 0
                while position < nbytes:
                    next_position = min(position + self.chunk_bytes, nbytes)
                    received = connection.recv_into(
                        staging_view[position:next_position]
                    )
                    if not received:
                        raise RuntimeError(
                            "TCP peer closed during tensor flow %d" % expected_flow
                        )
                    position += received
                received_bytes += nbytes
                stripe = offset // self.stripe_bytes
                if stripe not in locally_seen:
                    locally_seen.add(stripe)
                    with placed.get_lock():
                        should_place = not placed[stripe]
                        if should_place:
                            placed[stripe] = 1
                    if should_place:
                        destination[offset : offset + nbytes] = staging_view[:nbytes]
            end_ns = time.perf_counter_ns()
        if start_ns is None:
            start_ns = end_ns
        staging_view.release()
        return (
            transfer_id,
            received_bytes,
            start_ns,
            end_ns,
            end_ns,
        )

    def _receive_process_entry(
        self,
        result_pipe,
        listener,
        view,
        flow,
        prepared,
        placed,
    ):
        try:
            _set_affinity(self.cpus[flow])
            prepared.wait(timeout=self.timeout)
            result_pipe.send(
                (
                    True,
                    self._receive_dynamic_flow(
                        listener,
                        view,
                        flow,
                        placed,
                    ),
                )
            )
        except BaseException as error:
            result_pipe.send((False, "%s: %s" % (type(error).__name__, error)))
        finally:
            result_pipe.close()

    def _receive_processes(self, listeners, view, ready_event):
        context = multiprocessing.get_context("fork")
        prepared = context.Barrier(self.flows + 1)
        stripe_count = (len(view) + self.stripe_bytes - 1) // self.stripe_bytes
        placed = context.Array("b", stripe_count, lock=True)
        processes = []
        readers = []
        try:
            for flow in range(self.flows):
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(
                    target=self._receive_process_entry,
                    args=(
                        writer,
                        listeners[flow],
                        view,
                        flow,
                        prepared,
                        placed,
                    ),
                )
                process.start()
                writer.close()
                processes.append(process)
                readers.append(reader)
            prepared.wait(timeout=self.timeout)
            if ready_event is not None:
                ready_event.set()
            messages = [reader.recv() for reader in readers]
            for process in processes:
                process.join(timeout=self.timeout)
            failures = [message for succeeded, message in messages if not succeeded]
            if failures:
                raise RuntimeError("; ".join(failures))
            if not all(placed[stripe] for stripe in range(stripe_count)):
                raise RuntimeError("dynamic tensor workers left destination gaps")
            return [message for succeeded, message in messages]
        finally:
            for reader in readers:
                reader.close()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()

    def receive_into(
        self,
        tensor: Any,
        *,
        iterations: int = 1,
        ready_event: Optional[threading.Event] = None,
    ) -> ReceiveResult:
        """Receive repeated transfers into a writable contiguous CPU tensor."""

        buffer = _tensor_buffer(tensor, writable=True)
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if (
            self.worker_mode == "process"
            and not buffer.shared
            and not self.assume_shared
        ):
            raise ValueError(
                "process receivers require SharedTensor or shared PyTorch storage"
            )
        partitions = (
            None
            if self.worker_mode == "process"
            else _partitions(len(buffer.view), self.flows)
        )
        family = _family(self.local.address)
        listeners = []
        try:
            for flow in range(self.flows):
                listener = socket.socket(family, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                _bind_device(listener, self.local.interface)
                listener.settimeout(self.timeout)
                listener.bind(_address(self.local.address, self.base_port + flow))
                listener.listen(1)
                listeners.append(listener)
            if ready_event is not None and self.worker_mode == "thread":
                ready_event.set()
            if self.worker_mode == "process":
                results = self._receive_processes(
                    listeners,
                    buffer.view,
                    ready_event,
                )
            else:
                with ThreadPoolExecutor(max_workers=self.flows) as executor:
                    futures = [
                        executor.submit(
                            self._receive_flow,
                            listeners[flow],
                            buffer.view,
                            flow,
                            offset,
                            end,
                            iterations,
                        )
                        for flow, (offset, end) in enumerate(partitions)
                    ]
                    results = [future.result() for future in futures]
        finally:
            for listener in listeners:
                listener.close()
        transfer_ids = {result[0] for result in results}
        received_bytes = sum(result[1] for result in results)
        if len(transfer_ids) != 1 or received_bytes != len(buffer.view) * iterations:
            raise RuntimeError("received tensor flows belong to different transfers")
        active_results = [result for result in results if result[1] > 0]
        seconds = (
            max(result[3] for result in active_results)
            - min(result[2] for result in active_results)
        ) / 1e9
        completion_seconds = (
            max(result[4] for result in active_results)
            - min(result[2] for result in active_results)
        ) / 1e9
        return ReceiveResult(received_bytes, seconds, completion_seconds)


def transfer_tensor_local(
    source: Any,
    destination: Any,
    *,
    interfaces: Sequence[str] = ("eth1", "eth2"),
    addresses: Optional[Sequence[str]] = None,
    flows: int = 40,
    base_port: int = 6201,
    chunk_bytes: int = 1 << 20,
    stripe_bytes: int = 16 << 20,
    affinity: bool = True,
    verify_physical_path: bool = True,
    worker_mode: str = "auto",
    iterations: int = 1,
) -> LocalTransferResult:
    """Transfer a tensor through two local peer NICs and verify physical traffic.

    ``source`` and ``destination`` must be equally sized, contiguous CPU tensors.
    NumPy arrays and writable buffer-protocol objects work directly. CPU PyTorch
    tensors are supported through their zero-copy ``numpy()`` view. The function
    blocks until the destination is complete.
    """

    source_buffer = _tensor_buffer(source, writable=False)
    destination_buffer = _tensor_buffer(destination, writable=True)
    if len(source_buffer.view) != len(destination_buffer.view):
        raise ValueError("source and destination tensors must have equal byte sizes")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if worker_mode not in {"auto", "thread", "process"}:
        raise ValueError("worker_mode must be auto, thread, or process")
    selected_mode = (
        "process"
        if worker_mode == "auto" and destination_buffer.shared
        else "thread" if worker_mode == "auto" else worker_mode
    )
    if selected_mode == "process" and not destination_buffer.shared:
        raise ValueError("process transfers require a shared destination tensor")
    endpoints = benchmark._endpoints(interfaces, addresses)
    if len(endpoints) != 2:
        raise ValueError("local tensor transfer requires exactly two interfaces")
    sender_endpoint, receiver_endpoint = endpoints
    cpu_map = benchmark._allocate_cpus(endpoints, flows, affinity)
    sender = TensorSender(
        NetworkEndpoint(sender_endpoint.interface, sender_endpoint.address),
        receiver_endpoint.address,
        flows=flows,
        base_port=base_port,
        chunk_bytes=chunk_bytes,
        stripe_bytes=stripe_bytes,
        cpus=cpu_map[sender_endpoint.interface],
        worker_mode=selected_mode,
    )
    receiver = TensorReceiver(
        NetworkEndpoint(receiver_endpoint.interface, receiver_endpoint.address),
        flows=flows,
        base_port=base_port,
        chunk_bytes=chunk_bytes,
        stripe_bytes=stripe_bytes,
        cpus=cpu_map[receiver_endpoint.interface],
        worker_mode=selected_mode,
    )
    if selected_mode == "process":
        context = multiprocessing.get_context("fork")
        ready = context.Event()
        result_reader, result_writer = context.Pipe(duplex=False)

        def receive_process():
            try:
                result_writer.send(
                    (
                        True,
                        receiver.receive_into(
                            destination,
                            iterations=iterations,
                            ready_event=ready,
                        ),
                    )
                )
            except BaseException as error:
                ready.set()
                result_writer.send((False, "%s: %s" % (type(error).__name__, error)))
            finally:
                result_writer.close()

        receiver_worker = context.Process(target=receive_process)
        receiver_worker.start()
        result_writer.close()
    else:
        ready = threading.Event()
        receive_result = []
        receive_error = []

        def receive_thread():
            try:
                receive_result.append(
                    receiver.receive_into(
                        destination,
                        iterations=iterations,
                        ready_event=ready,
                    )
                )
            except BaseException as error:
                receive_error.append(error)
                ready.set()

        receiver_worker = threading.Thread(
            target=receive_thread, name="tensor-receiver"
        )
        receiver_worker.start()
    if not ready.wait(timeout=receiver.timeout):
        raise RuntimeError("tensor receiver did not become ready")
    if selected_mode == "thread" and receive_error:
        raise receive_error[0]

    tx_before = (
        benchmark._link_counters(sender_endpoint.interface, "tx")
        if verify_physical_path
        else None
    )
    rx_before = (
        benchmark._link_counters(receiver_endpoint.interface, "rx")
        if verify_physical_path
        else None
    )
    operation_start_ns = time.perf_counter_ns()
    try:
        send_result = sender.send(source, iterations=iterations)
    finally:
        receiver_worker.join(timeout=receiver.timeout)
    if receiver_worker.is_alive():
        if selected_mode == "process":
            receiver_worker.terminate()
            receiver_worker.join()
        raise RuntimeError("tensor receiver did not finish")
    if selected_mode == "process":
        succeeded, received = result_reader.recv()
        result_reader.close()
        if not succeeded:
            raise RuntimeError(received)
        receive_statistics = received
    else:
        if receive_error:
            raise receive_error[0]
        receive_statistics = receive_result[0] if receive_result else None
    if receive_statistics is None or receive_statistics.nbytes != send_result.nbytes:
        raise RuntimeError("tensor receiver reported an incomplete transfer")
    total_seconds = (time.perf_counter_ns() - operation_start_ns) / 1e9

    if verify_physical_path:
        tx_after = benchmark._link_counters(sender_endpoint.interface, "tx")
        rx_after = benchmark._link_counters(receiver_endpoint.interface, "rx")
        tx_wire_gbps = benchmark._wire_gbps(
            tx_before, tx_after, receive_statistics.seconds
        )
        rx_wire_gbps = benchmark._wire_gbps(
            rx_before, rx_after, receive_statistics.seconds
        )
        wire_gbps = min(tx_wire_gbps, rx_wire_gbps)
        difference = abs(tx_wire_gbps - rx_wire_gbps) / max(
            tx_wire_gbps, rx_wire_gbps, 1.0
        )
        path_verified = wire_gbps > 0 and difference <= 0.02
    else:
        tx_wire_gbps = 0.0
        rx_wire_gbps = 0.0
        wire_gbps = 0.0
        path_verified = False
    return LocalTransferResult(
        nbytes=send_result.nbytes,
        seconds=receive_statistics.seconds,
        completion_seconds=receive_statistics.completion_seconds,
        payload_gbps=receive_statistics.gbps,
        completion_gbps=receive_statistics.completion_gbps,
        total_seconds=total_seconds,
        total_gbps=send_result.nbytes * 8 / total_seconds / 1e9,
        wire_gbps=wire_gbps,
        tx_wire_gbps=tx_wire_gbps,
        rx_wire_gbps=rx_wire_gbps,
        retransmits=send_result.retransmits,
        path_verified=path_verified,
        flows=send_result.flows,
    )
