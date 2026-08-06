# pyre-ignore-all-errors[21]: Test dependency and experiment package.
"""Hardware-independent tests for the dual-NIC TCP benchmark."""

import json
from types import SimpleNamespace

import pytest
from experiments.tcp_multiflow import benchmark


def _endpoint(interface, address, speed=100.0, node=0):
    return benchmark.Endpoint(interface, address, speed, node)


def test_parse_cpu_list():
    assert benchmark._parse_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    with pytest.raises(ValueError, match="invalid CPU range"):
        benchmark._parse_cpu_list("4-2")


def test_addresses_accept_and_normalize_ipv6():
    assert benchmark._addresses("2001:0db8::1,2001:db8::2") == [
        "2001:db8::1",
        "2001:db8::2",
    ]


def test_interface_addresses_select_global_ipv4_and_ipv6(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_run_json",
        lambda command: [
            {
                "addr_info": [
                    {"family": "inet6", "scope": "link", "local": "fe80::1"},
                    {
                        "family": "inet6",
                        "scope": "global",
                        "local": "2001:0db8::1",
                    },
                    {"family": "inet", "scope": "global", "local": "192.0.2.1"},
                ]
            }
        ],
    )

    assert benchmark._interface_ip_addresses("eth1") == [
        "2001:db8::1",
        "192.0.2.1",
    ]


def test_allocate_cpus_prefers_unique_local_cpus(monkeypatch):
    endpoints = [
        _endpoint("eth1", "192.0.2.1", node=0),
        _endpoint("eth2", "198.51.100.1", node=1),
    ]
    pools = {"eth1": [0, 1, 2], "eth2": [1, 3, 4]}
    monkeypatch.setattr(
        benchmark, "_cpu_pool", lambda endpoint: pools[endpoint.interface]
    )

    assigned = benchmark._allocate_cpus(endpoints, 2, True)

    assert assigned == {"eth1": [0, 1], "eth2": [3, 4]}


def test_client_specs_use_one_process_and_port_per_flow():
    endpoints = [
        _endpoint("eth1", "192.0.2.1"),
        _endpoint("eth2", "198.51.100.1"),
    ]
    args = SimpleNamespace(
        server_addresses=["192.0.2.2", "198.51.100.2"],
        flows_per_nic=2,
        base_port=5201,
        iperf3="iperf3",
        duration=30,
        omit=5,
        length="1M",
        reverse=False,
    )
    cpus = {"eth1": [0, 1], "eth2": [2, 3]}

    specs = benchmark._client_specs(endpoints, args, cpus)

    assert len(specs) == 4
    assert [spec.port for spec in specs] == [5201, 5202, 5201, 5202]
    assert [spec.cpu for spec in specs] == [0, 1, 2, 3]
    assert specs[0].command[:9] == (
        "iperf3",
        "--version4",
        "--client",
        "192.0.2.2",
        "--bind",
        "192.0.2.1",
        "--bind-dev",
        "eth1",
        "--port",
    )


def test_ipv6_specs_select_iperf3_ipv6_mode():
    endpoints = [
        _endpoint("eth1", "2001:db8:1::1"),
        _endpoint("eth2", "2001:db8:2::1"),
    ]
    args = SimpleNamespace(
        server_addresses=["2001:db8:1::2", "2001:db8:2::2"],
        flows_per_nic=1,
        base_port=5201,
        iperf3="iperf3",
        duration=30,
        omit=5,
        length="1M",
        reverse=False,
    )
    cpus = {"eth1": [0], "eth2": [1]}

    specs = benchmark._client_specs(endpoints, args, cpus)

    assert all(spec.command[1] == "--version6" for spec in specs)


def test_flow_result_and_summary_use_receiver_goodput():
    endpoints = [
        _endpoint("eth1", "192.0.2.1"),
        _endpoint("eth2", "198.51.100.1"),
    ]
    documents = [
        {
            "end": {
                "sum_sent": {"bits_per_second": 101e9, "retransmits": 3},
                "sum_received": {"bits_per_second": 98e9, "bytes": 1000},
            }
        },
        {
            "end": {
                "sum_sent": {"bits_per_second": 100e9, "retransmits": 2},
                "sum_received": {"bits_per_second": 97e9, "bytes": 2000},
            }
        },
    ]
    specs = [
        benchmark.ProcessSpec("eth1", 0, 5201, 0, ("iperf3",)),
        benchmark.ProcessSpec("eth2", 0, 5201, 1, ("iperf3",)),
    ]

    results = [
        benchmark._flow_result(spec, document)
        for spec, document in zip(specs, documents)
    ]
    summary = benchmark._summarize(endpoints, results)

    assert summary["total_gbps"] == pytest.approx(195.0)
    assert summary["total_retransmits"] == 5
    assert summary["nics"]["eth1"]["gbps"] == pytest.approx(98.0)


def test_wire_rate_adds_preamble_and_interpacket_gap():
    before = benchmark.LinkCounters(bytes=1000, packets=10)
    after = benchmark.LinkCounters(bytes=2000, packets=20)

    assert benchmark._wire_gbps(before, after, 1e-6) == pytest.approx(9.6)


def test_local_mode_defaults_to_measured_line_rate_configuration():
    args = benchmark._parser().parse_args(["local"])

    assert args.interfaces == ["eth1", "eth2"]
    assert args.flows_per_nic == 24
    assert args.counter_delay == 7
    assert args.sample_seconds == 5
    assert args.target_gbps == 195.0


def test_client_run_writes_passing_summary(monkeypatch, tmp_path):
    endpoints = [
        _endpoint("eth1", "192.0.2.1"),
        _endpoint("eth2", "198.51.100.1"),
    ]
    monkeypatch.setattr(benchmark, "_check_iperf3", lambda executable: None)
    monkeypatch.setattr(
        benchmark,
        "_endpoints",
        lambda interfaces, addresses: endpoints,
    )
    monkeypatch.setattr(benchmark, "_check_routes", lambda local, remote: None)

    def write_results(specs, output_dir, dry_run):
        paths = []
        for spec in specs:
            path = output_dir / ("%s-%d.json" % (spec.interface, spec.flow))
            path.write_text(
                json.dumps(
                    {
                        "end": {
                            "sum_sent": {
                                "bits_per_second": 102e9,
                                "retransmits": 0,
                            },
                            "sum_received": {
                                "bits_per_second": 101e9,
                                "bytes": 1,
                            },
                        }
                    }
                )
            )
            paths.append(path)
        return paths

    monkeypatch.setattr(benchmark, "_run_processes", write_results)
    args = benchmark._parser().parse_args(
        [
            "client",
            "--server-addresses",
            "192.0.2.2,198.51.100.2",
            "--flows-per-nic",
            "1",
            "--output-dir",
            str(tmp_path),
            "--no-affinity",
        ]
    )

    assert benchmark.run_client(args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] is True
    assert summary["total_gbps"] == pytest.approx(202.0)
