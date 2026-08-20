from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import ibverbs as ib
import pytest


class FakeMR:
    def __init__(self, length=1 << 20):
        self.length = length
        self.sges = []

    def sge(self, length, offset=0):
        if offset < 0 or length < 0 or offset + length > self.length:
            raise ValueError("SGE range exceeds fake MR")
        sge = SimpleNamespace(length=length, offset=offset)
        self.sges.append(sge)
        return sge


class FakeWC:
    def __init__(self, wr_id, error=None):
        self.wr_id = wr_id
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error


class FakeCQ:
    def __init__(self, auto_complete=True):
        self.auto_complete = auto_complete
        self.ready = deque()
        self.qp = None
        self.closed = False

    def poll(self, count):
        completions = []
        while self.ready and len(completions) < count:
            completions.append(self.ready.popleft())
        if self.qp is not None:
            self.qp.provider_outstanding -= len(completions)
        return completions

    def close(self):
        self.closed = True


class FakeQP:
    _next_qp_num = 1

    def __init__(self, cq, pd=None, fail_post=None):
        self.cq = cq
        self.cq.qp = self
        self.pd = pd
        self.fail_post = fail_post
        self.post_calls = []
        self.provider_outstanding = 0
        self.max_outstanding = 0
        self.closed = False
        self.qp_num = self._next_qp_num
        FakeQP._next_qp_num += 1
        self.init_args = None
        self.rtr_args = None
        self.rts_args = None

    def post_send(self, wrs):
        wrs = list(wrs)
        if self.fail_post is not None:
            raise self.fail_post
        self.post_calls.append(wrs)
        self.provider_outstanding += len(wrs)
        self.max_outstanding = max(self.max_outstanding, self.provider_outstanding)
        if self.cq.auto_complete:
            self.cq.ready.extend(FakeWC(wr.wr_id) for wr in wrs)

    def to_init(self, *args, **kwargs):
        self.init_args = (args, kwargs)

    def to_rtr(self, *args, **kwargs):
        self.rtr_args = (args, kwargs)

    def to_rts(self, *args, **kwargs):
        self.rts_args = (args, kwargs)

    def close(self):
        self.closed = True


class FakeContext:
    def __init__(self, *, auto_complete=True, max_qp_rd_atom=16):
        self.auto_complete = auto_complete
        self.max_qp_rd_atom = max_qp_rd_atom
        self.cqs = []

    def create_cq(self, cqe):
        cq = FakeCQ(self.auto_complete)
        cq.cqe = cqe
        self.cqs.append(cq)
        return cq

    def query_device(self):
        return SimpleNamespace(max_qp_rd_atom=self.max_qp_rd_atom)

    def query_port(self, port):
        return SimpleNamespace(active_mtu=5, lid=1)


class FakePD:
    def __init__(self, context):
        self.context = context
        self.qps = []
        self.init_attrs = []

    def create_qp(self, init_attr):
        self.init_attrs.append(init_attr)
        qp = FakeQP(init_attr.send_cq, pd=self)
        self.qps.append(qp)
        return qp


class AdvancingClock:
    def __init__(self, step=0.1):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def make_scheduler(qp_count=2, queue_depth=2, chunk_size=4, auto_complete=True):
    cqs = [FakeCQ(auto_complete) for _ in range(qp_count)]
    qps = [FakeQP(cq) for cq in cqs]
    scheduler = ib.RdmaReadScheduler(
        qps,
        cqs,
        queue_depth=queue_depth,
        chunk_size=chunk_size,
    )
    return scheduler, qps, cqs


def test_read_many_stripes_chunks_and_continuously_refills_qps():
    scheduler, qps, _ = make_scheduler()
    mr = FakeMR(40)
    request = ib.RdmaReadRequest(
        mr,
        remote_addr=0x1000,
        rkey=17,
        length=40,
    )

    batch = scheduler.read_many([request])

    assert batch.done
    assert not batch.failed
    assert batch.completed_bytes == 40
    assert batch.completed_chunks == 10
    assert [sum(map(len, qp.post_calls)) for qp in qps] == [5, 5]
    assert [list(map(len, qp.post_calls)) for qp in qps] == [[2, 2, 1], [2, 2, 1]]
    assert all(qp.max_outstanding <= 2 for qp in qps)

    work_requests = [wr for qp in qps for call in qp.post_calls for wr in call]
    assert len({wr.wr_id for wr in work_requests}) == 10
    assert sorted(wr.sg_list[0].offset for wr in work_requests) == list(range(0, 40, 4))
    assert sorted(wr.remote_addr for wr in work_requests) == list(
        range(0x1000, 0x1000 + 40, 4)
    )
    assert all(wr.opcode == ib.WROpcode.RDMA_READ for wr in work_requests)
    assert all(wr.send_flags & ib.SendFlags.SIGNALED for wr in work_requests)


def test_multiple_requests_share_one_pipeline_and_flush_callback_runs_once():
    scheduler, _, _ = make_scheduler(qp_count=2, queue_depth=1, chunk_size=4)
    first = ib.RdmaReadRequest(FakeMR(12), 0x1000, 1, 12, tag="first")
    second = ib.RdmaReadRequest(FakeMR(8), 0x2000, 2, 8, tag="second")
    flushes = []

    batch = scheduler.read_many([first, second], on_complete=lambda: flushes.append(1))

    assert batch.completed_bytes == 20
    assert batch.completed_chunks == 5
    assert batch.requests == (first, second)
    assert flushes == [1]
    scheduler.progress(batch)
    assert flushes == [1]


def test_write_many_stripes_chunks_and_uses_write_opcode():
    scheduler, qps, _ = make_scheduler(qp_count=2, queue_depth=2, chunk_size=4)
    request = ib.RdmaWriteRequest(FakeMR(20), 0x3000, 9, 20)

    batch = scheduler.write_many([request])

    assert batch.completed_bytes == 20
    assert batch.completed_chunks == 5
    work_requests = [wr for qp in qps for call in qp.post_calls for wr in call]
    assert all(wr.opcode == ib.WROpcode.RDMA_WRITE for wr in work_requests)
    assert sorted(wr.remote_addr for wr in work_requests) == list(
        range(0x3000, 0x3000 + 20, 4)
    )


def test_async_wait_cooperatively_completes_a_batch():
    scheduler, _, _ = make_scheduler(qp_count=2, queue_depth=2, chunk_size=4)
    request = ib.RdmaReadRequest(FakeMR(24), 0x1000, 1, 24)

    batch = asyncio.run(scheduler.read_many_async([request]))

    assert batch.done
    assert batch.completed_bytes == 24


def test_cancelled_async_wait_poisons_and_closes_owned_resources():
    async def cancel_wait():
        context = FakeContext(auto_complete=False)
        pd = FakePD(context)
        scheduler = ib.RdmaReadScheduler.create(
            context,
            pd,
            qp_count=1,
            queue_depth=2,
            chunk_size=4,
        )
        batch = scheduler.submit(
            [ib.RdmaReadRequest(FakeMR(16), 0x1000, 1, 16)]
        )
        task = asyncio.create_task(scheduler.wait_async(batch, timeout=None))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return scheduler, batch, pd, context

    scheduler, batch, pd, context = asyncio.run(cancel_wait())

    assert scheduler.poisoned
    assert scheduler.closed
    assert batch.failed
    assert all(qp.closed for qp in pd.qps)
    assert all(cq.closed for cq in context.cqs)


def test_single_qp_scheduler_can_run_repeated_batches():
    scheduler, qps, _ = make_scheduler(qp_count=1, queue_depth=2, chunk_size=4)
    calls = []

    first = scheduler.read_many(
        [ib.RdmaReadRequest(FakeMR(12), 0x1000, 1, 12)],
        on_complete=lambda: calls.append("first"),
    )
    second = scheduler.read_many(
        [ib.RdmaReadRequest(FakeMR(8), 0x2000, 2, 8)],
        on_complete=lambda: calls.append("second"),
    )

    assert first.completed_chunks == 3
    assert second.completed_chunks == 2
    assert calls == ["first", "second"]
    work_requests = [wr for call in qps[0].post_calls for wr in call]
    assert len({wr.wr_id for wr in work_requests}) == 5


def test_create_allocates_owned_qps_and_cqs_on_shared_context_and_pd():
    context = FakeContext()
    pd = FakePD(context)

    scheduler = ib.RdmaReadScheduler.create(
        context,
        pd,
        qp_count=3,
        queue_depth=7,
        chunk_size=1024,
        cq_slack=2,
    )

    assert scheduler.context is context
    assert scheduler.pd is pd
    assert scheduler.qps == tuple(pd.qps)
    assert scheduler.cqs == tuple(context.cqs)
    assert [cq.cqe for cq in context.cqs] == [9, 9, 9]
    assert all(attr.max_send_wr == 9 for attr in pd.init_attrs)
    scheduler.close()
    assert all(qp.closed for qp in pd.qps)
    assert all(cq.closed for cq in context.cqs)
    assert not getattr(context, "closed", False)
    assert not getattr(pd, "closed", False)


def test_connect_raises_read_atomic_limits_to_device_capability():
    context = FakeContext(max_qp_rd_atom=8)
    pd = FakePD(context)
    scheduler = ib.RdmaReadScheduler.create(
        context,
        pd,
        qp_count=2,
        queue_depth=16,
    )
    remotes = [
        ib.QPInfo(qp_num=100 + index, psn=0, lid=1, gid=b"\x01" * 16, port=1, mtu=4)
        for index in range(2)
    ]

    scheduler.connect(remotes, port=1, sgid_index=3, access=0)

    for qp in pd.qps:
        assert qp.rtr_args[1]["mtu"] == 4
        assert qp.rtr_args[1]["max_dest_rd_atomic"] == 8
        assert qp.rts_args[1]["max_rd_atomic"] == 8


def test_timeout_poisons_scheduler_and_closes_owned_resources():
    context = FakeContext(auto_complete=False)
    pd = FakePD(context)
    scheduler = ib.RdmaReadScheduler.create(
        context,
        pd,
        qp_count=2,
        queue_depth=2,
        chunk_size=4,
        clock=AdvancingClock(),
    )
    request = ib.RdmaReadRequest(FakeMR(32), 0x1000, 1, 32)
    batch = scheduler.submit([request])

    with pytest.raises(ib.RdmaReadTimeout):
        scheduler.wait(batch, timeout=0.15)

    assert scheduler.poisoned
    assert scheduler.closed
    assert batch.failed
    assert all(qp.closed for qp in pd.qps)
    assert all(cq.closed for cq in context.cqs)


def test_bad_completion_is_routed_to_batch_and_poisons_scheduler():
    scheduler, _, cqs = make_scheduler(
        qp_count=1,
        queue_depth=1,
        chunk_size=4,
        auto_complete=False,
    )
    batch = scheduler.submit([ib.RdmaReadRequest(FakeMR(4), 0x1000, 1, 4)])
    wr_id = next(iter(scheduler._pending))
    cqs[0].ready.append(FakeWC(wr_id, error=ValueError("bad completion")))

    with pytest.raises(ib.RdmaReadError, match="bad completion") as caught:
        scheduler.progress(batch)

    assert isinstance(caught.value.__cause__, ValueError)
    assert scheduler.poisoned
    assert batch.failed
    with pytest.raises(RuntimeError, match="poisoned"):
        scheduler.submit([])


def test_unknown_completion_id_is_rejected():
    scheduler, _, cqs = make_scheduler(
        qp_count=1,
        queue_depth=1,
        chunk_size=4,
        auto_complete=False,
    )
    batch = scheduler.submit([ib.RdmaReadRequest(FakeMR(4), 0x1000, 1, 4)])
    cqs[0].ready.append(FakeWC(9999))

    with pytest.raises(ib.RdmaReadError, match="unknown wr_id"):
        scheduler.progress(batch)

    assert scheduler.poisoned


def test_post_failure_cleans_up_batch_state():
    cq = FakeCQ(auto_complete=False)
    qp = FakeQP(cq, fail_post=OSError("post failed"))
    scheduler = ib.RdmaReadScheduler([qp], [cq], queue_depth=1, chunk_size=4)

    with pytest.raises(ib.RdmaReadError, match="post failed"):
        scheduler.submit([ib.RdmaReadRequest(FakeMR(4), 0x1000, 1, 4)])

    assert scheduler.poisoned
    assert scheduler._active is None
    assert not scheduler._pending


def test_request_and_scheduler_validate_ranges_and_configuration():
    with pytest.raises(TypeError, match="local_mr"):
        ib.RdmaReadRequest(object(), 0, 0, 1)
    with pytest.raises(ValueError, match="length"):
        ib.RdmaReadRequest(FakeMR(), 0, 0, 0)
    with pytest.raises(ValueError, match="address space"):
        ib.RdmaReadRequest(FakeMR(), (1 << 64) - 1, 0, 2)
    with pytest.raises(ValueError, match="at least one QP"):
        ib.RdmaReadScheduler([], [])
    with pytest.raises(ValueError, match="chunk_size"):
        ib.RdmaReadScheduler([FakeQP(FakeCQ())], [FakeCQ()], chunk_size=1 << 32)


def test_empty_batch_completes_and_invokes_callback_without_posting():
    scheduler, qps, _ = make_scheduler()
    calls = []

    batch = scheduler.submit([], on_complete=lambda: calls.append("done"))

    assert batch.done
    assert batch.completed_bytes == 0
    assert calls == ["done"]
    assert not any(qp.post_calls for qp in qps)


@pytest.mark.integration
def test_multi_qp_scheduler_reads_over_real_loopback(ctx, dev_name, first_active):
    from _rc import HostBuffer
    from conftest import find_roce_gid

    _, port = first_active
    gid_index, gid = find_roce_gid(ctx, dev_name, port)
    port_attr = ctx.query_port(port)
    pd = ctx.alloc_pd()
    local = HostBuffer(pd, 64, fill=0)
    remote = HostBuffer(pd, 64, fill=0xA5)
    initiator = ib.RdmaReadScheduler.create(
        ctx,
        pd,
        qp_count=2,
        queue_depth=4,
        chunk_size=8,
    )
    responder = ib.RdmaReadScheduler.create(
        ctx,
        pd,
        qp_count=2,
        queue_depth=4,
        chunk_size=8,
    )
    access = ib.AccessFlags.LOCAL_WRITE | ib.AccessFlags.REMOTE_READ
    try:
        initiator.connect(
            responder.local_infos(port_attr, gid, port=port),
            port=port,
            sgid_index=gid_index,
            access=access,
        )
        responder.connect(
            initiator.local_infos(port_attr, gid, port=port),
            port=port,
            sgid_index=gid_index,
            access=access,
        )

        batch = initiator.read_many(
            [
                ib.RdmaReadRequest(
                    local.mr,
                    remote_addr=remote.addr,
                    rkey=remote.rkey,
                    length=64,
                )
            ]
        )

        assert batch.completed_chunks == 8
        assert local.get_bytes() == b"\xa5" * 64
    finally:
        initiator.close()
        responder.close()
        local.close()
        remote.close()
        pd.close()
