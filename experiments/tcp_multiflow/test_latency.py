# pyre-ignore-all-errors[21]: Test dependency and experiment package.
"""Hardware-independent tests for the TCP latency benchmark."""

import argparse

import pytest
from experiments.tcp_multiflow import benchmark, latency


def test_sizes_parse_suffixes_and_reject_duplicates():
    assert latency._sizes("1,64,4k,2M") == [1, 64, 4096, 2 << 20]

    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        latency._sizes("1K,1024")


def test_percentile_uses_nearest_rank():
    samples_ns = [1000, 2000, 3000, 4000, 5000]

    assert latency._percentile(samples_ns, 0.95) == 5.0
    assert latency._percentile(samples_ns, 0.50) == 3.0


def test_counter_delta():
    before = benchmark.LinkCounters(bytes=100, packets=2)
    after = benchmark.LinkCounters(bytes=350, packets=7)

    assert latency._counter_delta(before, after) == benchmark.LinkCounters(
        bytes=250, packets=5
    )


def test_latency_defaults_cover_small_and_large_messages():
    args = latency._parser().parse_args(["run"])

    assert args.interfaces == ["eth1", "eth2"]
    assert args.sizes == [1, 64, 1 << 10, 4 << 10, 64 << 10]
    assert args.iterations == 10000
