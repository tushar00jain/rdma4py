#!/usr/bin/env python3
"""Sweep TCP write sizes and flow counts over the local physical NIC path."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from experiments.tcp_multiflow import benchmark


def _flow_counts(value: str) -> List[int]:
    try:
        values = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "flow counts must be positive integers"
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("flow counts must be positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("flow counts must be unique")
    return values


def _lengths(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one TCP write size")
    if len({item.lower() for item in values}) != len(values):
        raise argparse.ArgumentTypeError("TCP write sizes must be unique")
    if any(
        not re.fullmatch(r"[1-9][0-9]*[KMG]?", item, re.IGNORECASE) for item in values
    ):
        raise argparse.ArgumentTypeError(
            "TCP write sizes must be positive bytes with an optional K/M/G suffix"
        )
    return values


def _path_component(length: str) -> str:
    return "length-%s" % length.lower()


def _tuned_result(results, target_gbps: float):
    verified = [result for result in results if result["path_verified"]]
    if not verified:
        return None
    at_target = [result for result in verified if result["wire_gbps"] >= target_gbps]
    if at_target:
        return min(at_target, key=lambda result: result["flow_count"])
    return max(verified, key=lambda result: result["wire_gbps"])


def _write_csv(path: Path, results):
    fields = [
        "write_size",
        "flow_count",
        "payload_gbps",
        "wire_gbps",
        "tx_wire_gbps",
        "rx_wire_gbps",
        "total_retransmits",
        "path_verified",
        "passed",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def run(args) -> int:
    if args.counter_delay + args.sample_seconds >= args.duration:
        raise RuntimeError(
            "counter delay plus sample seconds must be less than measured duration"
        )
    root = args.output_dir or benchmark._default_output_dir("sweep")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for length in args.lengths:
        for flows in args.flows:
            output_dir = root / _path_component(length) / ("flows-%02d" % flows)
            run_args = argparse.Namespace(
                interfaces=args.interfaces,
                addresses=args.addresses,
                flows_per_nic=flows,
                base_port=args.base_port,
                iperf3=args.iperf3,
                output_dir=output_dir,
                no_affinity=args.no_affinity,
                dry_run=args.dry_run,
                duration=args.duration,
                omit=args.omit,
                counter_delay=args.counter_delay,
                sample_seconds=args.sample_seconds,
                length=length,
                target_gbps=args.target_gbps,
                reverse=args.reverse,
            )
            print("\n[%s, %d flows]" % (length, flows))
            benchmark.run_local(run_args)
            if args.dry_run:
                continue
            result = json.loads((output_dir / "summary.json").read_text())
            result["write_size"] = length
            result["output_dir"] = str(output_dir)
            results.append(result)

    if args.dry_run:
        return 0

    tuned = {}
    for length in args.lengths:
        choice = _tuned_result(
            [result for result in results if result["write_size"] == length],
            args.target_gbps,
        )
        tuned[length] = choice
    report = {
        "interfaces": args.interfaces,
        "addresses": args.addresses,
        "direction": "reverse" if args.reverse else "forward",
        "target_gbps": args.target_gbps,
        "duration": args.duration,
        "omit": args.omit,
        "counter_delay": args.counter_delay,
        "sample_seconds": args.sample_seconds,
        "results": results,
        "tuned_by_write_size": tuned,
    }
    (root / "sweep.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(root / "sweep.csv", results)

    print("\nwrite size  tuned flows  payload  wire rate  target")
    path_verified = all(result["path_verified"] for result in results)
    for length, result in tuned.items():
        if result is None:
            print("%10s  no verified physical-path result" % length)
            continue
        print(
            "%10s  %11d  %7.2f  %9.2f  %s"
            % (
                length,
                result["flow_count"],
                result["payload_gbps"],
                result["wire_gbps"],
                "PASS" if result["wire_gbps"] >= args.target_gbps else "MISS",
            )
        )
    print("Results are in %s" % root)
    return 0 if path_verified else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interfaces",
        type=benchmark._csv,
        default=["eth1", "eth2"],
        help="sender and receiver interfaces (default: eth1,eth2)",
    )
    parser.add_argument(
        "--addresses",
        type=benchmark._addresses,
        help="sender and receiver addresses (default: auto-detect)",
    )
    parser.add_argument(
        "--lengths",
        type=_lengths,
        default=["1K", "4K", "16K", "64K", "256K", "1M"],
        help="comma-separated iperf3 write sizes (default: 1K,4K,16K,64K,256K,1M)",
    )
    parser.add_argument(
        "--flows",
        type=_flow_counts,
        default=[1, 2, 4, 8, 16, 24],
        help="comma-separated flow counts (default: 1,2,4,8,16,24)",
    )
    parser.add_argument("--duration", type=benchmark._positive_int, default=8)
    parser.add_argument("--omit", type=benchmark._nonnegative_int, default=1)
    parser.add_argument("--counter-delay", type=benchmark._nonnegative_int, default=3)
    parser.add_argument("--sample-seconds", type=benchmark._positive_int, default=3)
    parser.add_argument(
        "--target-gbps", type=benchmark._nonnegative_float, default=195.0
    )
    parser.add_argument("--base-port", type=benchmark._positive_int, default=5201)
    parser.add_argument("--iperf3", default="iperf3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--no-affinity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
