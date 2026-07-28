# pyre-ignore-all-errors[21]: Optional test dependencies.
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

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not hasattr(torch._C._distributed_c10d, "ReconfigureOptions"),
        reason="fault-tolerant all-reduce requires PyTorch nightly reconfiguration APIs",
    ),
]


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
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=%d" % len(gpus),
        "--module",
        "experiments.fault_tolerant_allreduce.benchmark",
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


def test_timeout_then_reconfigure_and_exact_allreduce():
    """Time out, reject reuse, reconfigure fresh QPs, and verify exact bytes."""

    gpus, _ = _mapping()
    if len(gpus) < 2:
        pytest.skip("all-reduce requires at least two GPU/HCA mappings")
    result = _run(
        [
            "--reconfigure-count",
            "2",
            "--timeout-smoke-test",
            "--torch-process-group-smoke-test",
        ],
        ranks=2,
    )
    assert result["timeout_recovery_test"] is True
    assert result["torch_process_group_test"] is True
    assert result["dtype_parity"]
    assert all(entry["bit_exact"] for entry in result["dtype_parity"])


def test_pytorch_fault_tolerance_membership_edges():
    """Exercise shrink, merge, abort recovery, and dense rank reassignment."""

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
    assert result["merge_reconfigure_test"] is True
    assert result["abort_reconfigure_test"] is True
    assert result["middle_shrink_reconfigure_test"] is True
    assert result["singleton_scale_reconfigure_test"] is True
