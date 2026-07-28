# pyre-ignore-all-errors[21]: Optional test dependencies and experiment package.
"""Hardware-independent tests for the optional all-reduce control plane."""

from __future__ import annotations

import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from experiments.fault_tolerant_allreduce import allreduce  # noqa: E402


def _payload(**overrides):
    value = {
        "stable_rank": 7,
        "nonce": "a" * 32,
        "host": "rank7.example",
        "port": 12345,
        "hca": "mlx5_0",
        "ib_port": 1,
        "gid_index": 3,
        "gid": "00" * 16,
        "max_bytes": 1 << 20,
        "qps": 8,
        "num_sms": 32,
    }
    value.update(overrides)
    return value


def test_reconfigure_handle_round_trip_and_validation():
    handle = allreduce._encode_handle(_payload())
    member = allreduce._decode_handle(handle)
    assert member.stable_rank == 7
    assert member.host == "rank7.example"
    assert member.qps == 8

    with pytest.raises(allreduce.ReconfigureError, match="invalid"):
        allreduce._decode_handle("not-an-allreduce-handle")
    with pytest.raises(allreduce.ReconfigureError, match="invalid"):
        allreduce._decode_handle(allreduce._encode_handle(_payload(gid="not a gid")))


@pytest.mark.parametrize(
    "overrides",
    [
        {"stable_rank": -1},
        {"nonce": ""},
        {"host": ""},
        {"port": 0},
        {"port": 65536},
        {"ib_port": 0},
        {"gid_index": -1},
        {"max_bytes": 0},
        {"qps": 0},
        {"num_sms": 0},
    ],
)
def test_reconfigure_handle_rejects_invalid_fields(overrides):
    with pytest.raises(allreduce.ReconfigureError, match="invalid values"):
        allreduce._decode_handle(allreduce._encode_handle(_payload(**overrides)))


def test_reconfigure_handle_rejects_schema_changes():
    missing = _payload()
    missing.pop("hca")
    with pytest.raises(allreduce.ReconfigureError, match="malformed"):
        allreduce._decode_handle(allreduce._encode_handle(missing))
    with pytest.raises(allreduce.ReconfigureError, match="malformed"):
        allreduce._decode_handle(
            allreduce._encode_handle({**_payload(), "unexpected": True})
        )


def test_reconfigure_frames_round_trip_and_reject_corruption():
    sender, receiver = socket.socketpair()
    try:
        allreduce._send_frame(sender, {"generation": 3, "members": [1, 2]})
        assert allreduce._recv_frame(receiver) == {
            "generation": 3,
            "members": [1, 2],
        }

        sender.sendall(allreduce._FRAME.pack(allreduce._MAX_FRAME + 1))
        with pytest.raises(allreduce.ReconfigureError, match="too large"):
            allreduce._recv_frame(receiver)

        sender.sendall(allreduce._FRAME.pack(1) + b"[")
        with pytest.raises(allreduce.ReconfigureError, match="malformed"):
            allreduce._recv_frame(receiver)

        sender.sendall(allreduce._FRAME.pack(2) + b"[]")
        with pytest.raises(allreduce.ReconfigureError, match="object"):
            allreduce._recv_frame(receiver)
    finally:
        sender.close()
        receiver.close()


def test_reconfigure_frame_rejects_truncation():
    sender, receiver = socket.socketpair()
    sender.sendall(allreduce._FRAME.pack(10) + b"short")
    sender.close()
    try:
        with pytest.raises(allreduce.ReconfigureError, match="peer closed"):
            allreduce._recv_frame(receiver)
    finally:
        receiver.close()


def _bare_group(**overrides):
    group = allreduce.ProcessGroup.__new__(allreduce.ProcessGroup)
    defaults = {
        "_closed": False,
        "_used_uuids": set(),
        "qps": 8,
        "num_sms": 8,
        "max_bytes": 1 << 20,
        "queue_depth": 256,
        "timeout": 1.0,
        "_rank": -1,
        "_size": 0,
        "uuid": None,
        "_failed": False,
        "_gpunetio": None,
        "_outgoing": [],
        "_incoming": [],
        "_next_buffers": None,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(group, name, value)
    group._handle = allreduce._encode_handle(
        _payload(
            stable_rank=0,
            nonce="local",
            qps=group.qps,
            num_sms=group.num_sms,
            max_bytes=group.max_bytes,
        )
    )
    return group


def _peer_handle(rank, **overrides):
    return allreduce._encode_handle(
        _payload(
            stable_rank=rank,
            nonce="peer-%d" % rank,
            port=12345 + rank,
            **overrides,
        )
    )


def test_single_rank_reconfigure_and_uuid_reuse():
    group = _bare_group()
    assert group._reconfigure(41, [group._handle]).wait()
    assert (group._rank, group._size, group.uuid, group._failed) == (
        0,
        1,
        41,
        False,
    )
    with pytest.raises(allreduce.ReconfigureError, match="fresh uuid"):
        group._reconfigure(41, [group._handle])


def test_identity_reconfigure_and_abort_recovery():
    group = _bare_group()
    group._reconfigure(51, [group._handle]).wait()
    group._reconfigure(52, [group._handle]).wait()
    assert (group._rank, group._size, group.uuid, group._failed) == (
        0,
        1,
        52,
        False,
    )
    group.abort()
    assert (group._rank, group._size, group.uuid, group._failed) == (
        -1,
        0,
        None,
        True,
    )
    group._reconfigure(53, [group._handle]).wait()
    assert (group._rank, group._size, group.uuid, group._failed) == (
        0,
        1,
        53,
        False,
    )


def test_nightly_process_group_reconfigure_contract():
    group = _bare_group()
    options = SimpleNamespace(
        uuid=71,
        handles=[group._handle],
        timeout=timedelta(seconds=2),
        hints={"test": "value"},
    )

    work = group.reconfigure(options)

    assert issubclass(allreduce.ProcessGroup, dist.ProcessGroup)
    assert isinstance(work, dist.Work)
    assert work.wait()
    assert group.supports_reconfigure
    assert group.getRank() == 0
    assert group.getSize() == 1
    assert group.getBackendName() == "ibverbs"


def test_process_group_options_and_registration():
    with pytest.raises(ValueError, match="hca"):
        allreduce.ProcessGroupOptions("")
    with pytest.raises(ValueError, match="max_bytes"):
        allreduce.ProcessGroupOptions("mlx5_0", max_bytes=0)

    if not hasattr(torch._C._distributed_c10d, "ReconfigureOptions"):
        with pytest.raises(RuntimeError, match="PyTorch nightly"):
            allreduce.register_backend("ibverbs_test_backend")
    else:
        name = "ibverbs_test_backend"
        assert allreduce.register_backend(name) == name
        assert dist.is_backend_available(name)
        assert allreduce.register_backend(name) == name


@pytest.mark.parametrize(
    ("handles", "message"),
    [
        ([], "at least one"),
        (["local", "local"], "unique"),
    ],
)
def test_reconfigure_rejects_bad_membership_shape(handles, message):
    group = _bare_group()
    values = [group._handle if value == "local" else value for value in handles]
    with pytest.raises(allreduce.ReconfigureError, match=message):
        group._reconfigure(1, values)


def test_reconfigure_rejects_missing_local_and_duplicate_stable_rank():
    group = _bare_group()
    with pytest.raises(allreduce.ReconfigureError, match="local handle"):
        group._reconfigure(1, [_peer_handle(1, qps=8, num_sms=8)])
    duplicate_rank = _peer_handle(0, qps=8, num_sms=8)
    with pytest.raises(allreduce.ReconfigureError, match="stable ranks"):
        group._reconfigure(2, [group._handle, duplicate_rank])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"qps": 9, "num_sms": 8}, "qps and num_sms"),
        ({"qps": 8, "num_sms": 9}, "qps and num_sms"),
        ({"qps": 8, "num_sms": 8, "max_bytes": 2 << 20}, "capacities"),
    ],
)
def test_reconfigure_rejects_incompatible_members(overrides, message):
    group = _bare_group()
    with pytest.raises(allreduce.ReconfigureError, match=message):
        group._reconfigure(1, [group._handle, _peer_handle(1, **overrides)])


@pytest.mark.parametrize(
    ("nbytes", "channels"),
    [(1, 1), (32767, 1), (32768, 1), (65535, 1), (65536, 2), (1 << 20, 32)],
)
def test_nccl_ring_simple_dynamic_channel_count(nbytes, channels):
    _, _, active = allreduce._nccl_channel_layout(nbytes // 4, 4, 32)
    assert active == channels


def test_completed_work_surface():
    work = allreduce.Work(3)
    assert work.is_completed()
    assert work.is_success()
    assert work.wait()
    assert work.result() == 3
    assert work.get_future().wait() == 3
    assert work.exception() is None
    assert isinstance(work, dist.Work)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_timeout_validation(value):
    with pytest.raises(ValueError, match="finite positive"):
        allreduce._seconds(value, 1.0)


def test_rc_timeout_exponent_is_bounded_and_monotonic():
    values = [allreduce._ibv_timeout_exponent(value, 3) for value in (1e-9, 1, 10)]
    assert values == sorted(values)
    assert all(0 <= value <= 31 for value in values)


@pytest.mark.parametrize(
    ("name", "width"),
    [
        ("float16", 2),
        ("bfloat16", 2),
        ("float32", 4),
        ("float64", 8),
        ("float8_e4m3fn", 1),
        ("float8_e5m2", 1),
        ("int8", 1),
        ("uint8", 1),
        ("int32", 4),
        ("uint32", 4),
        ("int64", 8),
        ("uint64", 8),
    ],
)
def test_nccl_sum_dtypes(name, width):
    class Tensor:
        dtype = name

        @staticmethod
        def element_size():
            return width

    assert allreduce._dtype_key(Tensor()) == (name, width)


def test_gpunetio_dtype_abi_codes_are_stable():
    assert allreduce._GPUNETIO_DTYPE_CODES == {
        "float16": 0,
        "bfloat16": 1,
        "float32": 2,
        "float64": 3,
        "int8": 4,
        "uint8": 5,
        "int32": 6,
        "uint32": 7,
        "int64": 8,
        "uint64": 9,
        "float8_e4m3fn": 10,
        "float8_e5m2": 11,
    }


class _FakeCudaFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeCuda:
    def __init__(self):
        self.memory = bytearray(64)
        self.freed = []
        self.cuMemAlloc_v2 = _FakeCudaFunction(self._alloc)
        self.cuMemFree_v2 = _FakeCudaFunction(self._free)
        self.cuMemcpyHtoD_v2 = _FakeCudaFunction(self._htod)
        self.cuMemcpyDtoH_v2 = _FakeCudaFunction(self._dtoh)
        self.cuMemsetD8_v2 = _FakeCudaFunction(self._memset)
        self.cuCtxGetDevice = _FakeCudaFunction(self._device)
        self.cuDeviceGetAttribute = _FakeCudaFunction(self._attribute)

    def _alloc(self, pointer, size):
        pointer._obj.value = 0x1000
        assert size <= len(self.memory)
        return 0

    def _free(self, pointer):
        self.freed.append(pointer)
        return 0

    def _htod(self, destination, source, size):
        assert destination == 0x1000
        self.memory[:size] = allreduce.ctypes.string_at(source, size)
        return 0

    def _dtoh(self, destination, source, size):
        assert source == 0x1000
        allreduce.ctypes.memmove(destination, bytes(self.memory[:size]), size)
        return 0

    def _memset(self, destination, value, size):
        assert destination == 0x1000
        self.memory[:size] = bytes([value]) * size
        return 0

    def _device(self, output):
        output._obj.value = 2
        return 0

    def _attribute(self, output, attribute, device):
        assert (attribute, device) == (13, 2)
        output._obj.value = 1_500_000
        return 0


def test_cuda_control_allocation(monkeypatch):
    cuda = _FakeCuda()
    monkeypatch.setattr(allreduce._CudaAllocation, "_lib", cuda)
    allocation = allreduce._CudaAllocation(16)
    allocation.write_u64([1, 2])
    assert allocation.read_i32(4) == [1, 0, 2, 0]
    allocation.zero()
    assert allocation.read_i32(4) == [0, 0, 0, 0]
    assert allocation.clock_rate_hz() == 1_500_000_000
    allocation.close()
    allocation.close()
    assert cuda.freed == [0x1000]


class _FakeTensor:
    dtype = "float32"
    is_cuda = True

    @staticmethod
    def data_ptr():
        return 0x1000

    @staticmethod
    def numel():
        return 4

    @staticmethod
    def element_size():
        return 4

    @staticmethod
    def is_contiguous():
        return True


def _fake_data_group(status_value):
    class Status:
        ptr = 0x3000

        @staticmethod
        def zero():
            return None

        @staticmethod
        def read_i32(count):
            assert count == 1
            return [status_value]

    pointer = type("Pointer", (), {"ptr": 0x4000})()
    mr = type("MR", (), {"addr": 0x5000, "lkey": 7})()
    state = type(
        "State",
        (),
        {
            "status": Status(),
            "outgoing_pointers": pointer,
            "incoming_pointers": pointer,
            "signal_mr": mr,
            "clock_rate_hz": 1_000_000_000,
        },
    )()
    group = allreduce.ProcessGroup.__new__(allreduce.ProcessGroup)
    group.work_buffer = _FakeTensor()
    group.scratch_buffer = type("Scratch", (), {"data_ptr": lambda self: 0x2000})()
    group._work_mr = type("WorkMR", (), {"lkey": 8})()
    group._gpunetio = state
    group._gpunetio_kernels = type(
        "Kernels", (), {"allreduce": lambda self, **kwargs: None}
    )()
    group._next_buffers = {
        "work_addr": 0x6000,
        "work_rkey": 9,
        "scratch_addr": 0x7000,
        "scratch_rkey": 10,
    }
    group._closed = False
    group._failed = False
    group.uuid = 1
    group._size = 2
    group._rank = 0
    group.max_bytes = 16
    group.timeout = 1.0
    group.num_sms = 1
    return group


def test_gpunetio_timeout_then_reconfigure_recovers(monkeypatch):
    data_group = _fake_data_group(-110)
    group = _bare_group(qps=1, num_sms=1)
    for name in (
        "work_buffer",
        "scratch_buffer",
        "_work_mr",
        "_gpunetio",
        "_gpunetio_kernels",
        "_next_buffers",
        "timeout",
        "uuid",
        "_rank",
        "_size",
    ):
        setattr(group, name, getattr(data_group, name))
    monkeypatch.setattr(allreduce._ibcuda, "synchronize", lambda: None)

    with pytest.raises(allreduce.AllReduceTimeoutError, match="GPUNetIO"):
        group._allreduce_tensor(_FakeTensor())
    assert group._failed
    with pytest.raises(allreduce.AllReduceError, match="failed"):
        group._allreduce_tensor(_FakeTensor())

    options = SimpleNamespace(
        uuid=2,
        handles=[group.get_reconfigure_handle()],
        timeout=timedelta(seconds=1),
        hints=None,
    )
    assert group.reconfigure(options).wait()
    assert (group.getRank(), group.getSize(), group._failed) == (0, 1, False)


def test_gpunetio_verbs_error_marks_group_failed(monkeypatch):
    group = _fake_data_group(-5)
    monkeypatch.setattr(allreduce._ibcuda, "synchronize", lambda: None)

    with pytest.raises(allreduce.AllReduceError, match="DOCA status -5"):
        group._allreduce_tensor(_FakeTensor())
    assert group._failed


def test_gpunetio_success(monkeypatch):
    group = _fake_data_group(0)
    monkeypatch.setattr(allreduce._ibcuda, "synchronize", lambda: None)

    tensor = _FakeTensor()
    options = torch._C._distributed_c10d.AllreduceOptions()
    work = group.allreduce([tensor], options)
    assert work.wait()
    assert work.result() == [tensor]
    assert not group._failed
