# GPUDirect RDMA benchmarks

Measured on 2026-07-21 with the `ibverbs` GPUDirect dma-buf path and torch
CUDA tensors. These are intra-host transfers through two physical ConnectX-7
ports on different NUMA domains, not CUDA copies over NVLink.

## Topology

Each GPU is paired with its nearest NIC. `nvidia-smi topo -m` reports `PIX`
between each GPU/NIC pair:

```text
NUMA 0                                              NUMA 1
GPU 0 (0000:1b:00.0)                               GPU 4 (0000:a3:00.0)
        | PIX, PCIe 5.0 x16                               | PIX, PCIe 5.0 x16
mlx5_0 (0000:18:00.0) ===== 400 Gb/s RoCE ===== mlx5_6 (0000:9a:00.0)
```

The host has two Intel Xeon Platinum 8480C sockets. GPU 0 and `mlx5_0` are on
NUMA node 0; GPU 4 and `mlx5_6` are on NUMA node 1. Both PCIe links negotiated
32.0 GT/s x16. The benchmark process was pinned to CPU 0 in the source HCA's
NUMA node. Although the two GPUs also have an NV18 path, it is not used by
these RDMA writes.

| Component | Version / configuration |
|---|---|
| GPUs | 2x NVIDIA H100 80GB HBM3 |
| GPU driver | 580.82.07 |
| HCAs | 2x ConnectX-7, firmware 28.38.1002 |
| HCA ports | Ethernet, RoCE v2, 400 Gb/s, MTU 4096 |
| HCA PCIe | 32.0 GT/s x16 on both endpoints |
| Kernel | Linux 6.16.1 |
| libibverbs | 61.0 |
| Python | 3.14.5 |
| torch / CUDA | torch 2.13.0+cu130 / CUDA 13.0 |
| Source revision | `823a56452630ffc0a683e05ce654985430318652` |

## Method

- Source and destination are separate contiguous `torch.uint8` CUDA tensors.
  Each tensor is registered against its local HCA's protection domain with
  `ibverbs.cuda.register_tensor` and the dma-buf path.
- The data operation is a one-way RC `RDMA_WRITE` from GPU 0 to GPU 4. Source
  CUDA work is synchronized before timing. GPUDirect writes are flushed and
  the destination pattern is checked after timing.
- Latency is the median and p99 host time from posting one signaled write to
  polling its successful sender completion. Writes are serialized on one QP.
  It includes Python/Cython posting and polling overhead, but not a CUDA flush
  or an application-level ping-pong.
- Bandwidth uses 16 RC QPs with queue depth 64. QPs write non-overlapping
  tensor regions; only the last request in each 64-write batch is signaled.
  The result is useful payload bytes divided by host wall time, reported as
  decimal GB/s and Gb/s. Each value is the median of five runs.
- Each size is warmed up first. Latency uses 10,000 iterations for small
  messages and at least 100 for large messages. Bandwidth targets 8 GiB per
  run, caps small-message runs at 1,048,576 writes, and uses at least four
  full batches per QP.

## Results

| Message size | p50 latency (us) | p99 latency (us) | Bandwidth (GB/s) | Bandwidth (Gb/s) |
|---:|---:|---:|---:|---:|
| 8 B | 5.022 | 6.705 | 0.114 | 0.91 |
| 64 B | 5.035 | 5.753 | 0.912 | 7.30 |
| 1 KiB | 5.135 | 5.897 | 14.026 | 112.21 |
| 4 KiB | 5.364 | 5.995 | 42.347 | 338.77 |
| 16 KiB | 5.842 | 6.464 | 46.910 | 375.28 |
| 64 KiB | 6.844 | 7.667 | 48.126 | 385.01 |
| 256 KiB | 13.046 | 14.368 | 48.470 | 387.76 |
| 1 MiB | 29.260 | 29.583 | **48.549** | **388.39** |
| 4 MiB | 94.247 | 95.107 | 48.525 | 388.20 |
| 16 MiB | 353.884 | 354.902 | 48.459 | 387.67 |
| 64 MiB | 1390.150 | 1594.362 | 48.433 | 387.47 |

Peak useful bandwidth is 48.55 GB/s, or 388.39 Gb/s. That is 97.1% of the
nominal 400 Gb/s port rate. Bandwidth is effectively saturated from 64 KiB
through 64 MiB.

### QP scaling

The full size sweep uses 16 QPs because it produced the highest median at
1 MiB. A single QP was already close to line rate, so the additional QPs
improved this workload by only 0.9%.

| QPs | Queue depth per QP | 1 MiB bandwidth (GB/s) | Bandwidth (Gb/s) |
|---:|---:|---:|---:|
| 1 | 64 | 48.130 | 385.04 |
| 2 | 64 | 48.369 | 386.95 |
| 4 | 64 | 48.478 | 387.82 |
| 8 | 64 | 48.529 | 388.23 |
| 16 | 64 | **48.548** | **388.38** |

## Reproduce

Run from the repository root after installing the package and torch into a
virtual environment. Torch must use its VMM-backed allocator so the tensors
can be exported as dma-buf file descriptors.

```bash
taskset -c 0 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python ibverbs/benchmarks/gpudirect.py \
  --src-gpu 0 --src-hca mlx5_0 \
  --dst-gpu 4 --dst-hca mlx5_6 \
  --qps 16
```

The script prints JSON containing the topology metadata, iteration counts,
and unrounded results. Change the GPU/HCA arguments only as pairs after
checking locality with `nvidia-smi topo -m`.

## CuTe all-reduce versus NCCL

Measured on GPU 0/`mlx5_0` and GPU 1/`mlx5_3` on 2026-07-27. Both
implementations used a rank-ordered Ring/Simple algorithm, a 4 MiB protocol
buffer, and the same 64-CTA ceiling. NCCL's P2P and shared-memory transports were
disabled, so its baseline also crossed the IB/RoCE network instead of NVLink.
The optional ibverbs implementation used the direct GPUNetIO transport with
one RC QP per channel (64 QPs). The timed tensors were typed views of its
registered work buffer, avoiding extra staging copies.

| Payload | ibverbs/CuTe latency | NCCL latency | ibverbs/CuTe alg. BW | NCCL alg. BW | Bit exact |
|---:|---:|---:|---:|---:|:---:|
| 1 MiB | 1.624 ms | **0.245 ms** | 0.65 GB/s | **4.28 GB/s** | ✅ |
| 16 MiB | **0.492 ms** | 0.653 ms | **34.10 GB/s** | 25.69 GB/s | ✅ |
| 64 MiB | **1.545 ms** | 1.677 ms | **43.43 GB/s** | 40.02 GB/s | ✅ |

The two-rank exact-byte matrix passed at 16 B, the three sizes on each side of
the 32 KiB and 64 KiB scheduling boundaries, and 1 MiB for float16, bfloat16,
float32, float64, float8 E4M3FN/E5M2, int8, uint8, int32, uint32, int64, and
uint64. This covers all 12 NCCL 2.29 datatypes on SM90, where NCCL enables FP8
reduction. Inputs include NaNs, infinities, signed zero, subnormals, and integer
overflow. A three-rank floating-point boundary run also passed before the
survivors rebuilt a two-rank ring after removing rank 2.

The missing-participant test uses a 50 ms GPU deadline. It verifies
`AllReduceTimeoutError`, rejection of further work on the poisoned group,
fresh-QP reconfiguration, and an exact post-recovery collective.

The benchmark preconditions the GPUs and NICs on the largest timed payload
before either implementation is measured, reducing idle-clock bias. The 1 MiB
case remains latency-bound by CuTe launch, GPU CQ polling, GPUDirect acquire
fencing, CTA barriers, and the host-visible status check. Payload dominates
those fixed costs at 16 MiB and above, where this GPUNetIO ring exceeds the
network-only NCCL baseline. The table records steady-state warmed medians, not
cold-call latency; short-message GPUNetIO results were also more variable.

### All-reduce channel scaling

For GPUNetIO, a QP must be owned by one channel so untagged receives cannot be
consumed by another independently scheduled CTA. The benchmark therefore
swept the CuTe SM/CTA count and the NCCL CTA ceiling together, with the same
number of QPs. Sixty-four was fastest at 64 MiB on this H100/ConnectX-7 pair.

| SM/CTAs and QPs | 64 MiB latency | Algorithm bandwidth |
|---:|---:|---:|
| 32 | 1.796 ms | 37.37 GB/s |
| 48 | 3.091 ms | 21.71 GB/s |
| 64 | **1.545 ms** | **43.43 GB/s** |

Reproduce the comparison (the script sets the NCCL controls before importing
torch):

```bash
torchrun --standalone --nproc-per-node=2 ibverbs/benchmarks/allreduce.py \
  --gpus 0,1 --hcas mlx5_0,mlx5_3 --sms 64 \
  --sizes 1m,16m,64m --warmups 10 --iterations 30 \
  --timeout-smoke-test
```
