# pyre-ignore-all-errors[21]: Test dependency and experiment package.
"""Hardware-independent tests for the TCP scaling sweep."""

import argparse
import json

import pytest
from experiments.tcp_multiflow import sweep


def test_sweep_values_are_validated():
    assert sweep._flow_counts("1,2,16") == [1, 2, 16]
    assert sweep._lengths("1K,64k,1M") == ["1K", "64k", "1M"]

    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        sweep._flow_counts("1,0")
    with pytest.raises(argparse.ArgumentTypeError, match="optional"):
        sweep._lengths("1.5K")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        sweep._lengths("1K,1k")


def test_tuning_uses_fewest_flows_that_reach_target():
    results = [
        {"flow_count": 1, "wire_gbps": 180.0, "path_verified": True},
        {"flow_count": 2, "wire_gbps": 196.0, "path_verified": True},
        {"flow_count": 4, "wire_gbps": 199.0, "path_verified": True},
    ]

    assert sweep._tuned_result(results, 195.0)["flow_count"] == 2
    assert sweep._tuned_result(results, 200.0)["flow_count"] == 4


def test_sweep_writes_aggregate_results(monkeypatch, tmp_path):
    rates = {1: 180.0, 2: 196.0, 4: 199.0}

    def run_local(args):
        args.output_dir.mkdir(parents=True)
        wire_gbps = rates[args.flows_per_nic]
        (args.output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "flow_count": args.flows_per_nic,
                    "payload_gbps": wire_gbps - 10,
                    "wire_gbps": wire_gbps,
                    "tx_wire_gbps": wire_gbps,
                    "rx_wire_gbps": wire_gbps,
                    "total_retransmits": 0,
                    "path_verified": True,
                    "passed": wire_gbps >= args.target_gbps,
                }
            )
        )
        return 0

    monkeypatch.setattr(sweep.benchmark, "run_local", run_local)
    args = sweep._parser().parse_args(
        [
            "--lengths",
            "4K",
            "--flows",
            "1,2,4",
            "--duration",
            "4",
            "--counter-delay",
            "1",
            "--sample-seconds",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert sweep.run(args) == 0
    report = json.loads((tmp_path / "sweep.json").read_text())
    assert report["tuned_by_write_size"]["4K"]["flow_count"] == 2
    assert len(report["results"]) == 3
    assert (tmp_path / "sweep.csv").is_file()
