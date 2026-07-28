# Fault-tolerant GPUNetIO all-reduce

This experiment is a pure-Python, one-process-per-GPU Ring/Simple all-reduce
for PyTorch nightly. `ProcessGroup` directly subclasses
`torch.distributed.ProcessGroup`; the data path uses only GPUNetIO-exported RC
QPs and a persistent CuTe kernel. It does not use a host-posted fallback, CUDA
IPC, NVLink, or NCCL. NCCL is used only by the benchmark as a performance and
exact-byte oracle.

The experiment is intentionally not part of the installable `ibverbs` package.
Run it from the repository root after installing the low-level bindings and
optional CuTe dependency:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
pip install -e './ibverbs[gpunetio-cutedsl]'
```

Use the PyTorch nightly index matching the installed CUDA toolkit if it is not
CUDA 13.0. Stable releases without c10d `ReconfigureOptions` are rejected. The
GPUNetIO bridge also requires the DOCA runtime, development headers, and device
bitcode described in the [ibverbs README](../../ibverbs/README.md).

## c10d integration

```python
import torch.distributed as dist
from experiments.fault_tolerant_allreduce import (
    ProcessGroupOptions,
    register_backend,
)

register_backend()
options = ProcessGroupOptions(
    hca=local_hca,
    gpu=local_rank,
    max_bytes=64 * 1024 * 1024,
    num_sms=64,
)
dist.init_process_group(
    "ibverbs",
    pg_options=options,
    enable_reconfigure=True,
)
handle = dist._get_reconfigure_handle()
# Exchange one handle per member, ordered by the desired dense rank.
dist._reconfigure(uuid=1, handles=handles).wait()
dist.all_reduce(cuda_tensor)
```

The operation is synchronous and returns a completed c10d `Work`. A GPU-side
`clock64()` deadline covers verbs completion and reduction. A timeout or verbs
failure poisons the current ring; the interrupted tensor is undefined. Recover
by exchanging the surviving membership handles and calling `_reconfigure`
with a fresh UUID, which creates fresh QPs.

By default, the group uses one QP per CuTe CTA. Set `qps >= num_sms` to reserve
more. The benchmark forces NCCL to the same CTA ceiling and selects
Ring/Simple with a 4 MiB protocol buffer so floating-point reduction order can
be compared exactly.

## Correctness and fault coverage

The exact-byte matrix covers NCCL 2.29's fp8 E4M3FN/E5M2, fp16, bf16, fp32,
fp64, int8/uint8, int32/uint32, and int64/uint64 reductions. Inputs include
NaNs, infinities, signed zero, subnormals, and integer overflow. Fault tests
cover timeout poisoning followed by reconfiguration, last- and middle-rank
removal, late join, abort recovery, dense-rank reassignment, singleton
scale-down/up, identity rebuilds, invalid membership, and reused UUIDs.

Run the hardware-independent tests with:

```bash
pytest -q experiments/fault_tolerant_allreduce/test_allreduce.py
```

Hardware tests require comma-separated rank mappings and skip unless PyTorch
nightly exposes the reconfiguration API:

```bash
RDMA4PY_ALLREDUCE_GPUS=0,1 \
RDMA4PY_ALLREDUCE_HCAS=mlx5_0,mlx5_3 \
RDMA4PY_GPUNETIO_BITCODE=/path/to/device.bc \
pytest -q experiments/fault_tolerant_allreduce/test_allreduce_hardware.py
```

## Benchmark

The benchmark disables NCCL's P2P and shared-memory paths by default, so the
reference also crosses IB/RoCE rather than NVLink. Run it as a module so the
top-level experiment package is importable in every worker:

```bash
torchrun --standalone --nproc-per-node=2 --module \
  experiments.fault_tolerant_allreduce.benchmark \
  --gpus 0,1 --hcas mlx5_0,mlx5_3 --sms 64 \
  --sizes 1m,16m,64m --warmups 10 --iterations 30 \
  --timeout-smoke-test
```

Results measured on H100/ConnectX-7 with 64 CTAs and 64 QPs:

| Payload | GPUNetIO latency | NCCL latency | GPUNetIO alg. BW | NCCL alg. BW | Bit exact |
|---:|---:|---:|---:|---:|:---:|
| 1 MiB | 1.624 ms | **0.245 ms** | 0.65 GB/s | **4.28 GB/s** | ✅ |
| 16 MiB | **0.492 ms** | 0.653 ms | **34.10 GB/s** | 25.69 GB/s | ✅ |
| 64 MiB | **1.545 ms** | 1.677 ms | **43.43 GB/s** | 40.02 GB/s | ✅ |

Short messages remain launch- and synchronization-bound. At larger payloads,
the network transfer dominates those fixed costs.
