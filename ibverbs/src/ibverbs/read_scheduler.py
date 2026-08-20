"""Pipelined multi-QP RDMA reads over registered host or GPU memory.

This module is an optional policy layer over the low-level verbs objects.  It
does not import torch or CUDA: a local buffer only needs to expose the same
``sge(length, offset=...)`` interface as :class:`~ibverbs.MR` and
``ibverbs.cuda.GpuMR``.  GPU callers are responsible for making source CUDA
writes visible before serving reads and for flushing destination GPUDirect
writes once after a completed batch.

The scheduler is deliberately caller-driven.  It owns no background thread;
``progress`` polls completion queues, while ``wait`` and ``wait_async`` drive
progress synchronously or cooperatively from an asyncio task.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ._ibverbs import QPInitAttr, SendWR
from .enums import QPType, SendFlags, WROpcode
from .helpers import local_qp_info, QPInfo

_MAX_SGE_LENGTH = (1 << 32) - 1
_MAX_WR_ID = (1 << 64) - 1


class RdmaReadError(RuntimeError):
    """Raised when a read batch cannot safely continue.

    Posting and completion failures poison the scheduler because outstanding
    work cannot be cancelled through the verbs API.  Recreate the QPs before
    submitting another batch.
    """


class RdmaReadTimeout(TimeoutError):
    """Raised when a read batch does not complete before its deadline."""


@dataclass(frozen=True)
class RdmaReadRequest:
    """Describe one local destination and one remote source byte range.

    Args:
        local_mr: Registered local memory exposing ``sge(length, offset=...)``.
            Both :class:`~ibverbs.MR` and ``ibverbs.cuda.GpuMR`` satisfy this
            interface.
        remote_addr: Base virtual address of the peer's registered memory.
        rkey: Remote key for the peer memory region.
        length: Number of bytes to read.
        local_offset: Byte offset into ``local_mr``.
        remote_offset: Byte offset from ``remote_addr``.
        tag: Optional caller-owned value retained with the request.
    """

    local_mr: Any
    remote_addr: int
    rkey: int
    length: int
    local_offset: int = 0
    remote_offset: int = 0
    tag: Any = None

    def __post_init__(self) -> None:
        remote_addr = int(self.remote_addr)
        rkey = int(self.rkey)
        length = int(self.length)
        local_offset = int(self.local_offset)
        remote_offset = int(self.remote_offset)
        if not hasattr(self.local_mr, "sge"):
            raise TypeError("local_mr must expose sge(length, offset=...)")
        if remote_addr < 0 or remote_addr > _MAX_WR_ID:
            raise ValueError("remote_addr must be between 0 and 2**64 - 1")
        if rkey < 0 or rkey > _MAX_SGE_LENGTH:
            raise ValueError("rkey must be between 0 and 2**32 - 1")
        if length <= 0:
            raise ValueError("length must be positive")
        if local_offset < 0:
            raise ValueError("local_offset must be non-negative")
        if remote_offset < 0:
            raise ValueError("remote_offset must be non-negative")
        if remote_addr + remote_offset + length > _MAX_WR_ID + 1:
            raise ValueError("remote byte range exceeds the 64-bit address space")
        object.__setattr__(self, "remote_addr", remote_addr)
        object.__setattr__(self, "rkey", rkey)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "local_offset", local_offset)
        object.__setattr__(self, "remote_offset", remote_offset)


@dataclass(frozen=True)
class _ReadChunk:
    request_index: int
    request: RdmaReadRequest
    offset: int
    length: int


@dataclass(frozen=True)
class _PendingRead:
    qp_index: int
    chunk: _ReadChunk


class RdmaReadBatch:
    """State and counters for one submitted group of RDMA reads.

    Instances are created by :meth:`RdmaReadScheduler.submit`.  A batch can be
    advanced manually with :meth:`RdmaReadScheduler.progress`, or waited on
    with the scheduler's synchronous or asynchronous wait methods.
    """

    def __init__(
        self,
        scheduler: "RdmaReadScheduler",
        requests: tuple[RdmaReadRequest, ...],
        total_chunks: int,
        on_complete: Callable[[], None] | None,
    ) -> None:
        self.requests = requests
        self.total_bytes = sum(request.length for request in requests)
        self.total_chunks = total_chunks
        self.completed_bytes = 0
        self.completed_chunks = 0
        self.error: BaseException | None = None
        self._scheduler = scheduler
        self._on_complete = on_complete
        self._callback_called = False
        self._done = False

    @property
    def done(self) -> bool:
        """Whether all reads completed or the batch failed."""
        return self._done

    @property
    def failed(self) -> bool:
        """Whether the batch finished with an exception."""
        return self.error is not None

    def raise_for_status(self) -> None:
        """Raise the exception that ended this batch, if any."""
        if self.error is not None:
            raise self.error


class RdmaReadScheduler:
    """Continuously pipeline a batch of RDMA reads over multiple RC QPs.

    Use :meth:`create` to allocate QPs and CQs that share an existing context
    and protection domain.  Alternatively, inject already-created QPs and
    their send CQs into the constructor.  Exactly one batch may be active at a
    time, and one task or thread must own calls to ``submit``/``progress``.

    Every chunk is signaled so completion ``wr_id`` values can be routed back
    to the correct request.  As completions free send-queue slots, new chunks
    are posted immediately, keeping up to ``queue_depth`` operations in flight
    on each QP.

    This class never performs a CUDA visibility flush.  Pass an ``on_complete``
    callback to :meth:`submit` or :meth:`read_many` to flush exactly once after
    every work completion in the batch has succeeded.
    """

    def __init__(
        self,
        qps: Sequence[Any],
        cqs: Sequence[Any],
        *,
        queue_depth: int = 64,
        chunk_size: int = 1 << 20,
        poll_batch: int | None = None,
        context: Any = None,
        pd: Any = None,
        owns_resources: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build a scheduler around connected or not-yet-connected RC QPs.

        ``qps`` and ``cqs`` must have equal non-zero length, with each CQ being
        the send CQ for the corresponding QP.  Externally supplied resources
        are not closed unless ``owns_resources`` is true.
        """
        qps = tuple(qps)
        cqs = tuple(cqs)
        queue_depth = int(queue_depth)
        chunk_size = int(chunk_size)
        if not qps:
            raise ValueError("at least one QP is required")
        if len(qps) != len(cqs):
            raise ValueError("qps and cqs must have the same length")
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        if chunk_size <= 0 or chunk_size > _MAX_SGE_LENGTH:
            raise ValueError("chunk_size must be between 1 and 2**32 - 1")
        if poll_batch is None:
            poll_batch = queue_depth
        poll_batch = int(poll_batch)
        if poll_batch <= 0:
            raise ValueError("poll_batch must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.qps = qps
        self.cqs = cqs
        self.queue_depth = queue_depth
        self.chunk_size = chunk_size
        self.poll_batch = poll_batch
        self.context = context
        self.pd = pd
        self._owns_resources = bool(owns_resources)
        self._clock = clock
        self._active: RdmaReadBatch | None = None
        self._chunk_iter: Any = None
        self._input_exhausted = True
        self._pending: dict[int, _PendingRead] = {}
        self._outstanding = [0] * len(qps)
        self._next_wr_id = 1
        self._closed = False
        self._poisoned = False

    @classmethod
    def create(
        cls,
        context: Any,
        pd: Any,
        *,
        qp_count: int = 1,
        queue_depth: int = 64,
        chunk_size: int = 1 << 20,
        poll_batch: int | None = None,
        cq_slack: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> "RdmaReadScheduler":
        """Allocate RC QPs/CQs sharing ``context`` and ``pd``.

        The scheduler owns the created QPs and CQs, but never closes the
        caller-owned context or protection domain.  The QPs remain in RESET;
        exchange :meth:`local_infos` with the peer and call :meth:`connect`.
        """
        qp_count = int(qp_count)
        queue_depth = int(queue_depth)
        chunk_size = int(chunk_size)
        cq_slack = int(cq_slack)
        if qp_count <= 0:
            raise ValueError("qp_count must be positive")
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        if chunk_size <= 0 or chunk_size > _MAX_SGE_LENGTH:
            raise ValueError("chunk_size must be between 1 and 2**32 - 1")
        if poll_batch is not None and int(poll_batch) <= 0:
            raise ValueError("poll_batch must be positive")
        if cq_slack < 0:
            raise ValueError("cq_slack must be non-negative")
        pd_context = getattr(pd, "context", context)
        if pd_context is not context:
            raise ValueError("pd must belong to context")
        qps = []
        cqs = []
        try:
            for _ in range(qp_count):
                cq = context.create_cq(int(queue_depth) + cq_slack)
                cqs.append(cq)
                qp = pd.create_qp(
                    QPInitAttr(
                        send_cq=cq,
                        recv_cq=cq,
                        qp_type=QPType.RC,
                        max_send_wr=int(queue_depth) + cq_slack,
                        max_recv_wr=max(1, cq_slack),
                    )
                )
                qps.append(qp)
        except BaseException:
            for qp in reversed(qps):
                qp.close()
            for cq in reversed(cqs):
                cq.close()
            raise
        return cls(
            qps,
            cqs,
            queue_depth=queue_depth,
            chunk_size=chunk_size,
            poll_batch=poll_batch,
            context=context,
            pd=pd,
            owns_resources=True,
            clock=clock,
        )

    @property
    def qp_count(self) -> int:
        """Number of QPs across which chunks are striped."""
        return len(self.qps)

    @property
    def poisoned(self) -> bool:
        """Whether an indeterminate in-flight failure made the QPs unsafe."""
        return self._poisoned

    @property
    def closed(self) -> bool:
        """Whether the scheduler has been closed."""
        return self._closed

    def local_infos(
        self,
        port_attr: Any,
        gid: Any,
        *,
        port: int,
        psns: int | Sequence[int] = 0,
    ) -> tuple[QPInfo, ...]:
        """Return one out-of-band :class:`QPInfo` for every local QP."""
        self._ensure_usable()
        values = self._per_qp_values(psns, "psns")
        return tuple(
            local_qp_info(qp, port_attr, gid, port=port, psn=psn)
            for qp, psn in zip(self.qps, values)
        )

    def connect(
        self,
        remote_infos: Sequence[QPInfo],
        *,
        port: int,
        sgid_index: int,
        access: int,
        local_psns: int | Sequence[int] = 0,
        mtu: int | None = None,
        max_rd_atomic: int | None = None,
        max_dest_rd_atomic: int | None = None,
        timeout: int = 14,
        retry_cnt: int = 7,
        rnr_retry: int = 7,
    ) -> None:
        """Connect every local RC QP to the corresponding peer QP.

        When the atomic limits are omitted they are raised from the raw verbs
        default of one to the smaller of ``queue_depth`` and the device's
        ``max_qp_rd_atom`` capability.  This permits multiple outstanding RDMA
        reads while respecting the provider limit.
        """
        self._ensure_usable()
        remote_infos = tuple(remote_infos)
        if len(remote_infos) != self.qp_count:
            raise ValueError("remote_infos must contain one QPInfo per local QP")
        psns = self._per_qp_values(local_psns, "local_psns")
        if self.context is None:
            self.context = getattr(getattr(self.qps[0], "pd", None), "context", None)
        if max_rd_atomic is None or max_dest_rd_atomic is None:
            limit = self.queue_depth
            if self.context is not None:
                attr = self.context.query_device()
                limit = min(limit, max(1, int(attr.max_qp_rd_atom)))
            if max_rd_atomic is None:
                max_rd_atomic = limit
            if max_dest_rd_atomic is None:
                max_dest_rd_atomic = limit
        max_rd_atomic = int(max_rd_atomic)
        max_dest_rd_atomic = int(max_dest_rd_atomic)
        if not 1 <= max_rd_atomic <= 255:
            raise ValueError("max_rd_atomic must be between 1 and 255")
        if not 1 <= max_dest_rd_atomic <= 255:
            raise ValueError("max_dest_rd_atomic must be between 1 and 255")

        try:
            for qp, remote, psn in zip(self.qps, remote_infos, psns):
                negotiated_mtu = mtu
                if negotiated_mtu is None:
                    local_mtu = qp.pd.context.query_port(port).active_mtu
                    negotiated_mtu = min(int(local_mtu), int(remote.mtu))
                qp.to_init(port, access)
                qp.to_rtr(
                    remote,
                    sgid_index=sgid_index,
                    mtu=negotiated_mtu,
                    max_dest_rd_atomic=max_dest_rd_atomic,
                )
                qp.to_rts(
                    psn,
                    timeout=timeout,
                    retry_cnt=retry_cnt,
                    rnr_retry=rnr_retry,
                    max_rd_atomic=max_rd_atomic,
                )
        except BaseException:
            self._poison_resources()
            raise

    def submit(
        self,
        requests: Iterable[RdmaReadRequest],
        *,
        on_complete: Callable[[], None] | None = None,
    ) -> RdmaReadBatch:
        """Submit a batch and fill every QP up to its configured depth.

        ``on_complete`` runs exactly once, after all successful completions.
        A GPU caller can use it to call ``flush_gpudirect_writes`` once for the
        entire state-dict or tensor batch instead of once per tensor.
        """
        self._ensure_usable()
        if self._active is not None:
            raise RuntimeError("another RDMA read batch is already active")
        if on_complete is not None and not callable(on_complete):
            raise TypeError("on_complete must be callable")
        converted = []
        for request in requests:
            if not isinstance(request, RdmaReadRequest):
                raise TypeError("requests must contain RdmaReadRequest objects")
            converted.append(request)
        request_tuple = tuple(converted)
        total_chunks = sum(
            math.ceil(request.length / self.chunk_size) for request in request_tuple
        )
        batch = RdmaReadBatch(self, request_tuple, total_chunks, on_complete)
        self._active = batch
        self._chunk_iter = self._iter_chunks(request_tuple)
        self._input_exhausted = total_chunks == 0
        if total_chunks == 0:
            self._finish_success(batch)
            return batch
        try:
            self._refill()
        except BaseException as error:
            self._fail_transport(batch, error, "posting RDMA reads failed")
            batch.raise_for_status()
        return batch

    def progress(self, batch: RdmaReadBatch) -> int:
        """Poll all CQs once, route completions, and refill freed QP slots.

        Returns the number of completions processed.  A completion or posting
        failure aborts the batch and poisons the scheduler.
        """
        self._validate_batch(batch)
        if batch.done:
            batch.raise_for_status()
            return 0
        completed = 0
        try:
            for qp_index, cq in enumerate(self.cqs):
                count = min(self.poll_batch, self._outstanding[qp_index])
                if count == 0:
                    continue
                for wc in cq.poll(count):
                    pending = self._pending.pop(int(wc.wr_id), None)
                    if pending is None:
                        raise RdmaReadError(
                            "completion has unknown wr_id %d" % int(wc.wr_id)
                        )
                    if pending.qp_index != qp_index:
                        raise RdmaReadError(
                            "completion wr_id %d arrived on the wrong CQ"
                            % int(wc.wr_id)
                        )
                    self._outstanding[qp_index] -= 1
                    wc.raise_for_status()
                    batch.completed_chunks += 1
                    batch.completed_bytes += pending.chunk.length
                    completed += 1
            if completed:
                self._refill()
            if self._input_exhausted and not self._pending:
                self._finish_success(batch)
        except BaseException as error:
            self._fail_transport(batch, error, "RDMA read progress failed")
            batch.raise_for_status()
        return completed

    def wait(
        self,
        batch: RdmaReadBatch,
        *,
        timeout: float | None = 30.0,
        idle_sleep: float = 0.0,
    ) -> RdmaReadBatch:
        """Drive progress synchronously until ``batch`` completes.

        A timeout poisons the scheduler and destroys resources created by
        :meth:`create`, causing the provider to flush outstanding work.
        """
        self._validate_wait_options(timeout, idle_sleep)
        deadline = None if timeout is None else self._clock() + float(timeout)
        try:
            while not batch.done:
                completed = self.progress(batch)
                if batch.done:
                    break
                if deadline is not None and self._clock() >= deadline:
                    error = RdmaReadTimeout(
                        "timed out after %d of %d RDMA reads completed"
                        % (batch.completed_chunks, batch.total_chunks)
                    )
                    self._fail_transport(batch, error, None)
                    batch.raise_for_status()
                if completed == 0 and idle_sleep:
                    time.sleep(idle_sleep)
        except BaseException as error:
            if not batch.done:
                self._fail_transport(batch, error, None)
            raise
        batch.raise_for_status()
        return batch

    async def wait_async(
        self,
        batch: RdmaReadBatch,
        *,
        timeout: float | None = 30.0,
        idle_sleep: float = 0.0,
    ) -> RdmaReadBatch:
        """Cooperatively drive progress from an asyncio task."""
        self._validate_wait_options(timeout, idle_sleep)
        deadline = None if timeout is None else self._clock() + float(timeout)
        try:
            while not batch.done:
                self.progress(batch)
                if batch.done:
                    break
                if deadline is not None and self._clock() >= deadline:
                    error = RdmaReadTimeout(
                        "timed out after %d of %d RDMA reads completed"
                        % (batch.completed_chunks, batch.total_chunks)
                    )
                    self._fail_transport(batch, error, None)
                    batch.raise_for_status()
                await asyncio.sleep(idle_sleep)
        except BaseException as error:
            if not batch.done:
                self._fail_transport(batch, error, None)
            raise
        batch.raise_for_status()
        return batch

    def read_many(
        self,
        requests: Iterable[RdmaReadRequest],
        *,
        timeout: float | None = 30.0,
        idle_sleep: float = 0.0,
        on_complete: Callable[[], None] | None = None,
    ) -> RdmaReadBatch:
        """Submit and synchronously wait for a complete read batch."""
        return self.wait(
            self.submit(requests, on_complete=on_complete),
            timeout=timeout,
            idle_sleep=idle_sleep,
        )

    async def read_many_async(
        self,
        requests: Iterable[RdmaReadRequest],
        *,
        timeout: float | None = 30.0,
        idle_sleep: float = 0.0,
        on_complete: Callable[[], None] | None = None,
    ) -> RdmaReadBatch:
        """Submit and asynchronously wait for a complete read batch."""
        return await self.wait_async(
            self.submit(requests, on_complete=on_complete),
            timeout=timeout,
            idle_sleep=idle_sleep,
        )

    def close(self) -> None:
        """Close scheduler-owned QPs/CQs and make the scheduler unusable."""
        if self._closed:
            return
        if self._active is not None and not self._active.done:
            error = RdmaReadError("RDMA read scheduler closed with work in flight")
            self._active.error = error
            self._active._done = True
        self._active = None
        self._pending.clear()
        self._outstanding = [0] * self.qp_count
        self._closed = True
        self._close_owned(suppress=False)

    def __enter__(self) -> "RdmaReadScheduler":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def _iter_chunks(
        self, requests: tuple[RdmaReadRequest, ...]
    ) -> Iterable[_ReadChunk]:
        for request_index, request in enumerate(requests):
            for offset in range(0, request.length, self.chunk_size):
                yield _ReadChunk(
                    request_index=request_index,
                    request=request,
                    offset=offset,
                    length=min(self.chunk_size, request.length - offset),
                )

    def _refill(self) -> None:
        posts: list[list[tuple[int, _ReadChunk, SendWR]]] = [
            [] for _ in range(self.qp_count)
        ]
        made_progress = True
        while not self._input_exhausted and made_progress:
            made_progress = False
            for qp_index in range(self.qp_count):
                if (
                    self._outstanding[qp_index] + len(posts[qp_index])
                    >= self.queue_depth
                ):
                    continue
                try:
                    chunk = next(self._chunk_iter)
                except StopIteration:
                    self._input_exhausted = True
                    break
                wr_id = self._allocate_wr_id()
                request = chunk.request
                local_offset = request.local_offset + chunk.offset
                remote_addr = request.remote_addr + request.remote_offset + chunk.offset
                wr = SendWR(
                    wr_id=wr_id,
                    sg_list=[request.local_mr.sge(chunk.length, offset=local_offset)],
                    opcode=WROpcode.RDMA_READ,
                    send_flags=SendFlags.SIGNALED,
                    remote_addr=remote_addr,
                    rkey=request.rkey,
                )
                posts[qp_index].append((wr_id, chunk, wr))
                made_progress = True

        for qp_index, queued in enumerate(posts):
            if not queued:
                continue
            for wr_id, chunk, _ in queued:
                self._pending[wr_id] = _PendingRead(qp_index, chunk)
            self.qps[qp_index].post_send([wr for _, _, wr in queued])
            self._outstanding[qp_index] += len(queued)

    def _allocate_wr_id(self) -> int:
        for _ in range(_MAX_WR_ID):
            wr_id = self._next_wr_id
            self._next_wr_id = 1 if wr_id == _MAX_WR_ID else wr_id + 1
            if wr_id not in self._pending:
                return wr_id
        raise RdmaReadError("no free RDMA work-request IDs")

    def _finish_success(self, batch: RdmaReadBatch) -> None:
        batch._done = True
        self._active = None
        self._chunk_iter = None
        if batch._on_complete is not None and not batch._callback_called:
            batch._callback_called = True
            try:
                batch._on_complete()
            except BaseException as error:
                batch.error = error

    def _fail_transport(
        self,
        batch: RdmaReadBatch,
        error: BaseException,
        message: str | None,
    ) -> None:
        if message is not None and not isinstance(error, (RdmaReadError, TimeoutError)):
            wrapped = RdmaReadError("%s: %s" % (message, error))
            wrapped.__cause__ = error
            error = wrapped
        batch.error = error
        batch._done = True
        self._active = None
        self._chunk_iter = None
        self._input_exhausted = True
        self._pending.clear()
        self._outstanding = [0] * self.qp_count
        self._poisoned = True
        if self._owns_resources:
            self._closed = True
            self._close_owned(suppress=True)

    def _poison_resources(self) -> None:
        self._poisoned = True
        if self._owns_resources:
            self._closed = True
            self._close_owned(suppress=True)

    def _close_owned(self, *, suppress: bool) -> None:
        if not self._owns_resources:
            return
        first_error = None
        for resource in reversed(self.qps):
            try:
                resource.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        for resource in reversed(self.cqs):
            try:
                resource.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._owns_resources = False
        if first_error is not None and not suppress:
            raise first_error

    def _ensure_usable(self) -> None:
        if self._closed:
            raise RuntimeError("RDMA read scheduler is closed")
        if self._poisoned:
            raise RuntimeError("RDMA read scheduler is poisoned; recreate its QPs")

    def _validate_batch(self, batch: RdmaReadBatch) -> None:
        if not isinstance(batch, RdmaReadBatch) or batch._scheduler is not self:
            raise ValueError("batch does not belong to this scheduler")
        if not batch.done and batch is not self._active:
            raise ValueError("batch is not active")

    @staticmethod
    def _validate_wait_options(timeout: float | None, idle_sleep: float) -> None:
        if timeout is not None and float(timeout) <= 0:
            raise ValueError("timeout must be positive or None")
        if float(idle_sleep) < 0:
            raise ValueError("idle_sleep must be non-negative")

    def _per_qp_values(self, values: int | Sequence[int], name: str) -> tuple[int, ...]:
        if isinstance(values, int):
            return (int(values),) * self.qp_count
        converted = tuple(int(value) for value in values)
        if len(converted) != self.qp_count:
            raise ValueError("%s must contain one value per QP" % name)
        return converted


__all__ = [
    "RdmaReadBatch",
    "RdmaReadError",
    "RdmaReadRequest",
    "RdmaReadScheduler",
    "RdmaReadTimeout",
]
