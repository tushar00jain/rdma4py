#!/usr/bin/env python3
"""Benchmark striped TCP transfers between process-shared CPU tensors."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from experiments.tcp_multiflow import benchmark, python_benchmark, transport


def _fill_pattern(view: memoryview) -> None:
    block = bytes(range(251)) * ((1 << 20) // 251)
    position = 0
    while position < len(view):
        end = min(position + len(block), len(view))
        view[position:end] = block[: end - position]
        position = end


def _equal(left: memoryview, right: memoryview) -> bool:
    chunk_bytes = 16 << 20
    return all(
        left[offset : offset + chunk_bytes] == right[offset : offset + chunk_bytes]
        for offset in range(0, len(left), chunk_bytes)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interfaces", type=benchmark._csv, default=["eth1", "eth2"])
    parser.add_argument("--addresses", type=benchmark._addresses)
    parser.add_argument("--tensor-bytes", type=python_benchmark._size, default=64 << 20)
    parser.add_argument("--iterations", type=benchmark._positive_int, default=1024)
    parser.add_argument("--flows", type=benchmark._positive_int, default=40)
    parser.add_argument("--chunk-bytes", type=python_benchmark._size, default=1 << 20)
    parser.add_argument("--stripe-bytes", type=python_benchmark._size, default=16 << 20)
    parser.add_argument("--base-port", type=benchmark._positive_int, default=6201)
    parser.add_argument("--target-gbps", type=float, default=195.0)
    parser.add_argument("--no-affinity", action="store_true")
    parser.add_argument("--no-verify-path", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_storage = transport.SharedTensor(args.tensor_bytes)
    destination_storage = transport.SharedTensor(args.tensor_bytes)
    source = source_storage.numpy((args.tensor_bytes,))
    destination = destination_storage.numpy((args.tensor_bytes,))
    try:
        _fill_pattern(memoryview(source))
        result = transport.transfer_tensor_local(
            source,
            destination,
            interfaces=args.interfaces,
            addresses=args.addresses,
            flows=args.flows,
            base_port=args.base_port,
            chunk_bytes=args.chunk_bytes,
            stripe_bytes=args.stripe_bytes,
            affinity=not args.no_affinity,
            verify_physical_path=not args.no_verify_path,
            iterations=args.iterations,
        )
        verified = _equal(memoryview(source), memoryview(destination))
        report = {
            **asdict(result),
            "tensor_bytes": args.tensor_bytes,
            "iterations": args.iterations,
            "tensor_verified": verified,
            "target_gbps": args.target_gbps,
            "target_achieved": verified
            and result.path_verified
            and result.wire_gbps >= args.target_gbps,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        return 0 if report["target_achieved"] else 2
    finally:
        del source
        del destination
        source_storage.close()
        destination_storage.close()


if __name__ == "__main__":
    raise SystemExit(main())
