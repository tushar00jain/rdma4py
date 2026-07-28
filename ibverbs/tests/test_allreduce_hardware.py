"""Opt-in multi-GPU correctness and fault-recovery tests for all-reduce.

Set ``RDMA4PY_ALLREDUCE_GPUS`` and ``RDMA4PY_ALLREDUCE_HCAS`` to matching
comma-separated rank mappings. The tests remain explicit because choosing the
wrong GPU/HCA pair can turn a useful GPUDirect test into a topology benchmark.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _mapping():
    gpus = [
        value
        for value in os.environ.get("RDMA4PY_ALLREDUCE_GPUS", "").split(",")
        if value
    ]
    hcas = [
        value
        for value in os.environ.get("RDMA4PY_ALLREDUCE_HCAS", "").split(",")
        if value
    ]
    if not gpus or len(gpus) != len(hcas):
        pytest.skip("set matching RDMA4PY_ALLREDUCE_GPUS and RDMA4PY_ALLREDUCE_HCAS")
    return gpus, hcas


def _run(extra, *, ranks=None):
    gpus, hcas = _mapping()
    if ranks is not None:
        gpus = gpus[:ranks]
        hcas = hcas[:ranks]
    benchmark = Path(__file__).parents[1] / "benchmarks" / "allreduce.py"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=%d" % len(gpus),
        str(benchmark),
        "--gpus",
        ",".join(gpus),
        "--hcas",
        ",".join(hcas),
        "--sms",
        os.environ.get("RDMA4PY_ALLREDUCE_SMS", "64"),
        "--sizes",
        "16",
        "--warmups",
        "0",
        "--iterations",
        "1",
        *extra,
    ]
    bitcode = os.environ.get("RDMA4PY_GPUNETIO_BITCODE")
    if bitcode:
        command.extend(("--gpunetio-bitcode", bitcode))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        env=os.environ.copy(),
    )
    if completed.returncode:
        pytest.fail(
            "all-reduce subprocess failed\nstdout:\n%s\nstderr:\n%s"
            % (completed.stdout, completed.stderr)
        )
    start = completed.stdout.find("{\n")
    assert start >= 0, completed.stdout
    return json.loads(completed.stdout[start:])


def test_nccl_edge_matrix_timeout_and_same_membership_recovery():
    """Check exact bytes, timeout poisoning, and recovery on fresh QPs."""

    gpus, _ = _mapping()
    if len(gpus) < 2:
        pytest.skip("all-reduce requires at least two GPU/HCA mappings")
    result = _run(["--timeout-smoke-test"], ranks=2)
    assert result["timeout_recovery_test"] is True
    assert result["dtype_parity"]
    assert all(entry["bit_exact"] for entry in result["dtype_parity"])


def test_survivors_reconfigure_after_rank_removal():
    """Drop one rank, rebuild the ring, and verify the survivor result."""

    gpus, _ = _mapping()
    if len(gpus) < 3:
        pytest.skip("rank-removal recovery requires at least three mappings")
    result = _run(
        [
            "--parity-sizes",
            "16,65536",
            "--parity-dtypes",
            "float16,bfloat16,float32,float64",
            "--shrink-smoke-test",
        ]
    )
    assert result["shrink_reconfigure_test"] is True
