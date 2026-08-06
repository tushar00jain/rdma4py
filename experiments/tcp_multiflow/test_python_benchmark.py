# pyre-ignore-all-errors[21]: Test dependency and experiment package.
"""Hardware-independent tests for the multi-process Python TCP benchmark."""

import argparse
import json
import struct
from types import SimpleNamespace

import pytest
from experiments.tcp_multiflow import benchmark, python_benchmark


def _endpoint(interface, address, node):
    return benchmark.Endpoint(interface, address, 200.0, node)


def test_size_parses_binary_suffixes():
    assert python_benchmark._size("1") == 1
    assert python_benchmark._size("4K") == 4096
    assert python_benchmark._size("2m") == 2 << 20

    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        python_benchmark._size("0")


def test_worker_specs_use_one_python_process_per_flow():
    sender = _endpoint("eth1", "2001:db8::1", 0)
    receiver = _endpoint("eth2", "2001:db8::2", 1)
    args = SimpleNamespace(
        flows_per_nic=2,
        base_port=5201,
        length=4096,
        duration=8,
        omit=1,
    )
    cpus = {"eth1": [0, 1], "eth2": [2, 3]}

    servers, clients = python_benchmark._worker_specs(sender, receiver, args, cpus)

    assert [spec.port for spec in servers] == [5201, 5202]
    assert [spec.cpu for spec in servers] == [2, 3]
    assert [spec.cpu for spec in clients] == [0, 1]
    assert clients[0].command[1:4] == (
        "-m",
        "experiments.tcp_multiflow.python_benchmark",
        "client",
    )
    assert clients[0].command[-8:] == (
        "--port",
        "5201",
        "--length",
        "4096",
        "--duration",
        "8",
        "--omit",
        "1",
    )


def test_total_retransmits_reads_linux_tcp_info():
    info = bytearray(104)
    struct.pack_into("=I", info, 100, 17)
    sock = SimpleNamespace(getsockopt=lambda *args: bytes(info))

    assert python_benchmark._total_retransmits(sock) == 17


def test_flow_results_calculate_python_goodput(tmp_path):
    spec = benchmark.ProcessSpec("eth1", 0, 5201, 3, ("python",))
    result_path = tmp_path / "flow.json"
    result_path.write_text(
        json.dumps(
            {"bytes_sent": 25_000_000_000, "elapsed_seconds": 2, "retransmits": 4}
        )
    )

    results = python_benchmark._flow_results([spec], [result_path])

    assert results[0]["gbps"] == 100.0
    assert results[0]["retransmits"] == 4


def test_local_defaults_use_python_write_size():
    args = python_benchmark._parser().parse_args(["local"])

    assert args.interfaces == ["eth1", "eth2"]
    assert args.flows_per_nic == 24
    assert args.length == 1 << 20
