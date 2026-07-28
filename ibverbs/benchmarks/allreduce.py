#!/usr/bin/env python3
"""Compare the optional ibverbs/CuTe ring all-reduce with NCCL.

Launch one process per GPU.  This benchmark intentionally configures NCCL for
the Ring/Simple path, disables NVLink and shared-memory transports, fixes the
same CTA/SM ceiling used by the CuTe kernels, and compares raw result bytes.
NCCL is a benchmark/reference dependency only; ``ibverbs.allreduce`` never
imports or calls it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import statistics
import time
from pathlib import Path


def _early_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sms", type=int, default=32)
    parser.add_argument("--allow-nvlink-nccl", action="store_true")
    return parser.parse_known_args()[0]


_early = _early_args()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "Simple")
os.environ.setdefault("NCCL_MIN_CTAS", str(_early.sms))
os.environ.setdefault("NCCL_MAX_CTAS", str(_early.sms))
os.environ.setdefault("NCCL_MIN_NCHANNELS", str(_early.sms))
os.environ.setdefault("NCCL_MAX_NCHANNELS", str(_early.sms))
os.environ.setdefault("NCCL_NTHREADS", "512")
os.environ.setdefault("NCCL_BUFFSIZE", str(4 * 1024 * 1024))
if not _early.allow_nvlink_nccl:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    os.environ.setdefault("NCCL_NET", "IB")

import ibverbs as ib  # noqa: E402
from ibverbs.allreduce import (  # noqa: E402
    AllReduceError,
    AllReduceTimeoutError,
    ProcessGroup,
)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "uint32": torch.uint32,
    "int64": torch.int64,
    "uint64": torch.uint64,
}
_NCCL_DTYPES = {
    "int8": 0,
    "uint8": 1,
    "int32": 2,
    "uint32": 3,
    "int64": 4,
    "uint64": 5,
    "float16": 6,
    "float32": 7,
    "float64": 8,
    "bfloat16": 9,
    "float8_e4m3fn": 10,
    "float8_e5m2": 11,
}


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


class _NcclReference:
    """Small benchmark-only NCCL ABI wrapper, including unsigned dtypes."""

    def __init__(self, rank: int, world: int):
        path = Path(torch.__file__).parents[1] / "nvidia/nccl/lib/libnccl.so.2"
        self.lib = ctypes.CDLL(str(path) if path.is_file() else "libnccl.so.2")
        self.lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
        self.lib.ncclGetUniqueId.restype = ctypes.c_int
        self.lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            _NcclUniqueId,
            ctypes.c_int,
        ]
        self.lib.ncclCommInitRank.restype = ctypes.c_int
        self.lib.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.ncclAllReduce.restype = ctypes.c_int
        self.lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self.lib.ncclCommDestroy.restype = ctypes.c_int
        self.lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p

        unique_id = _NcclUniqueId()
        if rank == 0:
            self._check("ncclGetUniqueId", self.lib.ncclGetUniqueId(unique_id))
            payload = bytes(unique_id)
        else:
            payload = None
        objects = [payload]
        dist.broadcast_object_list(objects, src=0)
        ctypes.memmove(
            ctypes.addressof(unique_id), objects[0], ctypes.sizeof(unique_id)
        )
        self.comm = ctypes.c_void_p()
        self._check(
            "ncclCommInitRank",
            self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, unique_id, rank),
        )

    def _check(self, function: str, status: int):
        if status != 0:
            message = self.lib.ncclGetErrorString(status).decode(errors="replace")
            raise RuntimeError("%s failed: %s" % (function, message))

    def allreduce(self, tensor, dtype_name: str):
        stream = torch.cuda.current_stream(tensor.device).cuda_stream
        self._check(
            "ncclAllReduce",
            self.lib.ncclAllReduce(
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_void_p(tensor.data_ptr()),
                tensor.numel(),
                _NCCL_DTYPES[dtype_name],
                0,
                self.comm,
                ctypes.c_void_p(stream),
            ),
        )
        torch.cuda.synchronize(tensor.device)

    def close(self):
        if self.comm.value:
            self._check("ncclCommDestroy", self.lib.ncclCommDestroy(self.comm))
            self.comm = ctypes.c_void_p()


def _find_gid(context, device_name: str, port: int):
    base = "/sys/class/infiniband/%s/ports/%d/gid_attrs/types" % (
        device_name,
        port,
    )
    best = None
    attrs = context.query_port(port)
    for index in range(attrs.gid_tbl_len):
        gid = context.query_gid(port, index)
        if gid.raw == b"\0" * 16:
            continue
        try:
            with open(os.path.join(base, str(index))) as stream:
                gid_type = stream.read().strip().lower()
        except OSError:
            gid_type = ""
        link_local = gid.raw[0] == 0xFE and (gid.raw[1] & 0xC0) == 0x80
        score = (4 if "v2" in gid_type else 0) + (0 if link_local else 2)
        if best is None or score > best[0]:
            best = (score, index)
    if best is None:
        raise RuntimeError("no usable GID on %s port %d" % (device_name, port))
    return best[1]


def _parse_sizes(value: str):
    suffixes = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    sizes = []
    for item in value.split(","):
        item = item.strip().lower()
        multiplier = suffixes.get(item[-1], 1)
        if multiplier != 1:
            item = item[:-1]
        sizes.append(int(item) * multiplier)
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive")
    return sizes


def _parse_dtypes(value: str):
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in _DTYPES]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            "dtypes must be selected from %s" % ",".join(_DTYPES)
        )
    return names


_FLOAT_BITS = {
    torch.float8_e4m3fn: (
        torch.uint8,
        [0x00, 0x80, 0x7E, 0xFE, 0x7F, 0xFF, 0x01, 0x07],
    ),
    torch.float8_e5m2: (
        torch.uint8,
        [0x00, 0x80, 0x7C, 0xFC, 0x7D, 0xFF, 0x01, 0x03],
    ),
    torch.float16: (
        torch.uint16,
        [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x7FFF, 0x0001, 0x03FF],
    ),
    torch.bfloat16: (
        torch.uint16,
        [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7FFF, 0x0001, 0x007F],
    ),
    torch.float32: (
        torch.uint32,
        [
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7FFFFFFF,
            0x00000001,
            0x007FFFFF,
        ],
    ),
    torch.float64: (
        torch.uint64,
        [
            0x0000000000000000,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000000001,
            0x7FFFFFFFFFFFFFFF,
            0x0000000000000001,
            0x000FFFFFFFFFFFFF,
        ],
    ),
}


def _input(count: int, dtype, gpu: int, seed: int, rank: int):
    """Generate random data with arithmetic edge cases in the first elements."""

    generator = torch.Generator(device=gpu).manual_seed(seed)
    if dtype.is_floating_point:
        random_dtype = (
            torch.float32
            if dtype in {torch.float8_e4m3fn, torch.float8_e5m2}
            else dtype
        )
        result = torch.randn(
            count, dtype=random_dtype, device=gpu, generator=generator
        ).to(dtype)
        storage_dtype, bits = _FLOAT_BITS[dtype]
        shift = rank % len(bits)
        bits = bits[-shift:] + bits[:-shift] if shift else bits
        edge = torch.tensor(bits, dtype=storage_dtype, device=gpu).view(dtype)
    else:
        result = torch.randint(
            0 if dtype in {torch.uint8, torch.uint32, torch.uint64} else -8,
            9,
            (count,),
            dtype=dtype,
            device=gpu,
            generator=generator,
        )
        limits = torch.iinfo(dtype)
        values = [limits.min, limits.max, 0, 1]
        if limits.min < 0:
            values.append(-1)
        shift = rank % len(values)
        values = values[-shift:] + values[:-shift] if shift else values
        edge = torch.tensor(values, dtype=dtype, device=gpu)
    length = min(count, edge.numel())
    if length:
        result[:length] = edge[:length]
    return result


def _byte_mismatches(left, right) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def _median_seconds(call, warmups: int, iterations: int):
    for _ in range(warmups):
        call()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e9)
    return statistics.median(samples)


def _max_across_ranks(value: float, device: int) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def run(args):
    if not dist.is_available() or not torch.cuda.is_available():
        raise RuntimeError("CUDA torch.distributed is required for the benchmark")
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    gpu_ids = args.gpus or list(range(world))
    hcas = args.hcas.split(",")
    if len(gpu_ids) != world or len(hcas) != world:
        raise ValueError("--gpus and --hcas need exactly WORLD_SIZE entries")
    gpu = gpu_ids[rank]
    hca = hcas[rank]
    torch.cuda.set_device(gpu)
    nccl_reference = _NcclReference(rank, world)

    devices = {device.name: device for device in ib.get_device_list()}
    if hca not in devices:
        raise RuntimeError("RDMA device not found: %s" % hca)
    context = devices[hca].open()
    pd = context.alloc_pd()
    capacity = max(max(args.sizes), max(args.parity_sizes))
    work_buffer = torch.empty(capacity, dtype=torch.uint8, device=gpu)
    scratch_buffer = torch.empty(capacity, dtype=torch.uint8, device=gpu)
    gid_index = _find_gid(context, hca, args.port)
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    advertise_host = args.advertise_host or (
        "127.0.0.1" if local_world == world else socket.getfqdn()
    )
    group = ProcessGroup(
        context,
        pd,
        work_buffer,
        scratch_buffer,
        stable_rank=rank,
        gid_index=gid_index,
        port=args.port,
        advertise_host=advertise_host,
        qps=args.qps,
        num_sms=args.sms,
        timeout=args.timeout,
        queue_depth=args.queue_depth,
        transport=args.transport,
        gpu=gpu,
        gpunetio_arch=args.gpunetio_arch,
        gpunetio_bitcode=args.gpunetio_bitcode,
        nic_handler=args.nic_handler,
    )
    handles = [None] * world
    dist.all_gather_object(handles, group.get_reconfigure_handle())
    for generation in range(args.reconfigure_count):
        group.reconfigure(args.uuid + generation, handles, timeout=args.timeout).wait()
    dist.barrier()

    results = []
    parity = []
    timeout_recovery_verified = None
    shrink_verified = None
    try:
        # Compare storage bytes, not values: this catches NaN payload and
        # signed-zero differences as well as ordinary arithmetic mismatches.
        for name in args.parity_dtypes:
            dtype = _DTYPES[name]
            itemsize = torch.empty((), dtype=dtype).element_size()
            for parity_size in args.parity_sizes:
                count = parity_size // itemsize
                original = _input(count, dtype, gpu, args.seed + rank, rank)
                custom = work_buffer[: count * itemsize].view(dtype)
                custom.copy_(original)
                reference = original.clone()
                group.allreduce(custom)
                nccl_reference.allreduce(reference, name)
                mismatches = _byte_mismatches(custom, reference)
                mismatch_tensor = torch.tensor(mismatches, device=gpu)
                dist.all_reduce(mismatch_tensor, op=dist.ReduceOp.SUM)
                mismatches = int(mismatch_tensor.item())
                exact = mismatches == 0
                diagnostic = None
                if not exact:
                    different = torch.nonzero(
                        custom.view(torch.uint8) != reference.view(torch.uint8),
                        as_tuple=False,
                    ).flatten()
                    first = torch.tensor(
                        int(different[0].item()) if different.numel() else parity_size,
                        dtype=torch.int64,
                        device=gpu,
                    )
                    dist.all_reduce(first, op=dist.ReduceOp.MIN)
                    element_start = (int(first.item()) // itemsize) * itemsize
                    source = original.view(torch.uint8)[
                        element_start : element_start + itemsize
                    ]
                    sources = [torch.empty_like(source) for _ in range(world)]
                    dist.all_gather(sources, source)
                    scratch_source = scratch_buffer[
                        element_start : element_start + itemsize
                    ]
                    scratch_sources = [
                        torch.empty_like(scratch_source) for _ in range(world)
                    ]
                    dist.all_gather(scratch_sources, scratch_source)
                    diagnostic = {
                        "first_mismatched_byte": int(first.item()),
                        "first_mismatched_bytes_local": [
                            int(value) for value in different[:32].cpu().tolist()
                        ],
                        "input_hex_by_rank": [
                            bytes(value.cpu().tolist()).hex() for value in sources
                        ],
                        "scratch_hex_by_rank": [
                            bytes(value.cpu().tolist()).hex()
                            for value in scratch_sources
                        ],
                        "ibverbs_hex": bytes(
                            custom.view(torch.uint8)[
                                element_start : element_start + itemsize
                            ]
                            .cpu()
                            .tolist()
                        ).hex(),
                        "nccl_hex": bytes(
                            reference.view(torch.uint8)[
                                element_start : element_start + itemsize
                            ]
                            .cpu()
                            .tolist()
                        ).hex(),
                    }
                if rank == 0:
                    entry = {
                        "dtype": name,
                        "bytes_per_rank": count * itemsize,
                        "bit_exact": exact,
                        "mismatched_bytes": mismatches,
                    }
                    if diagnostic is not None:
                        entry["diagnostic"] = diagnostic
                    parity.append(entry)
                if not exact and args.require_bit_exact:
                    raise AssertionError(
                        "ibverbs result differs from pinned NCCL for %s at %d "
                        "bytes in %d result bytes"
                        % (name, count * itemsize, mismatches)
                    )

        if any(size % 4 for size in args.sizes):
            raise ValueError("benchmark sizes must be multiples of four")

        # Short GPUNetIO kernels can otherwise be dominated by an H100's
        # transition from its idle clock. Precondition both GPUs and NICs on
        # the largest timed payload (capped at 64 MiB) before measuring either
        # implementation. This work is outside every reported interval.
        precondition_bytes = min(max(args.sizes), 64 * 1024 * 1024)
        precondition = work_buffer[:precondition_bytes].view(torch.float32)
        precondition.zero_()
        for _ in range(args.precondition_iterations):
            group.allreduce(precondition)
        torch.cuda.synchronize(gpu)

        for size in args.sizes:
            count = size // 4
            original = _input(count, torch.float32, gpu, args.seed + rank, rank)
            custom = work_buffer[:size].view(torch.float32)
            custom.copy_(original)
            reference = original.clone()

            # Correctness is tested before timing.  The NCCL environment above
            # fixes the ring, protocol, CTA count, and chunk size required for
            # a meaningful bitwise comparison.
            group.allreduce(custom)
            dist.all_reduce(reference)
            torch.cuda.synchronize(gpu)
            mismatch = _byte_mismatches(custom, reference)
            bit_exact = mismatch == 0
            if not bit_exact and args.require_bit_exact:
                raise AssertionError(
                    "ibverbs result differs from pinned NCCL in %d bytes" % mismatch
                )

            def custom_call():
                custom.copy_(original)
                group.allreduce(custom)

            def nccl_call():
                reference.copy_(original)
                dist.all_reduce(reference)
                torch.cuda.synchronize(gpu)

            dist.barrier()
            custom_seconds = _median_seconds(custom_call, args.warmups, args.iterations)
            dist.barrier()
            nccl_seconds = _median_seconds(nccl_call, args.warmups, args.iterations)
            custom_seconds = _max_across_ranks(custom_seconds, gpu)
            nccl_seconds = _max_across_ranks(nccl_seconds, gpu)
            if rank == 0:
                results.append(
                    {
                        "bytes": size,
                        "bit_exact": bit_exact,
                        "ibverbs_ms": custom_seconds * 1e3,
                        "nccl_ms": nccl_seconds * 1e3,
                        "ibverbs_algorithm_GBps": size / custom_seconds / 1e9,
                        "nccl_algorithm_GBps": size / nccl_seconds / 1e9,
                        "ibverbs_bus_GBps": (2 * (world - 1) / world)
                        * size
                        / custom_seconds
                        / 1e9,
                        "nccl_bus_GBps": (2 * (world - 1) / world)
                        * size
                        / nccl_seconds
                        / 1e9,
                    }
                )
        if args.timeout_smoke_test:
            dist.barrier()
            local_verified = 1
            if rank == 0:
                probe = work_buffer[: 64 * 1024].view(torch.float32)
                probe.zero_()
                try:
                    group.allreduce(probe, timeout=args.timeout_smoke_seconds)
                except AllReduceTimeoutError:
                    pass
                else:
                    local_verified = 0
            else:
                time.sleep(max(1.0, 3 * args.timeout_smoke_seconds))
            dist.barrier()
            verified_tensor = torch.tensor(local_verified, device=gpu)
            dist.all_reduce(verified_tensor, op=dist.ReduceOp.MIN)
            timeout_verified = bool(verified_tensor.item())
            if not timeout_verified:
                raise AssertionError("GPUNetIO timeout smoke test did not time out")

            if rank == 0:
                try:
                    group.allreduce(probe)
                except AllReduceError:
                    pass
                else:
                    raise AssertionError("timed-out communicator remained reusable")
            dist.barrier()

            recovery_uuid = args.uuid + args.reconfigure_count
            group.reconfigure(recovery_uuid, handles, timeout=args.timeout).wait()
            recovery_count = (64 * 1024) // 4
            original = _input(
                recovery_count, torch.float32, gpu, args.seed + rank + 101, rank
            )
            custom = work_buffer[: 64 * 1024].view(torch.float32)
            custom.copy_(original)
            reference = original.clone()
            group.allreduce(custom)
            nccl_reference.allreduce(reference, "float32")
            recovery_mismatch = torch.tensor(
                _byte_mismatches(custom, reference), device=gpu
            )
            dist.all_reduce(recovery_mismatch, op=dist.ReduceOp.SUM)
            timeout_recovery_verified = bool(
                timeout_verified and recovery_mismatch.item() == 0
            )
            if not timeout_recovery_verified:
                raise AssertionError("communicator did not recover after timeout")

        if args.shrink_smoke_test:
            if world < 3:
                raise ValueError("--shrink-smoke-test requires at least three ranks")
            survivor_count = world - 1
            survivor_ranks = list(range(survivor_count))
            survivor_group = dist.new_group(ranks=survivor_ranks, backend="nccl")
            dist.barrier()
            local_shrink_verified = 1
            if rank < survivor_count:
                generation_offset = args.reconfigure_count + int(
                    args.timeout_smoke_test
                )
                group.reconfigure(
                    args.uuid + generation_offset,
                    handles[:survivor_count],
                    timeout=args.timeout,
                ).wait()
                shrink_count = (64 * 1024) // 4
                original = _input(
                    shrink_count,
                    torch.float32,
                    gpu,
                    args.seed + rank + 202,
                    rank,
                )
                custom = work_buffer[: 64 * 1024].view(torch.float32)
                custom.copy_(original)
                reference = original.clone()
                group.allreduce(custom)
                dist.all_reduce(reference, group=survivor_group)
                torch.cuda.synchronize(gpu)
                mismatch = torch.tensor(_byte_mismatches(custom, reference), device=gpu)
                dist.all_reduce(mismatch, group=survivor_group)
                local_shrink_verified = int(mismatch.item() == 0)
            dist.barrier()
            verified_tensor = torch.tensor(local_shrink_verified, device=gpu)
            dist.all_reduce(verified_tensor, op=dist.ReduceOp.MIN)
            shrink_verified = bool(verified_tensor.item())
            if not shrink_verified:
                raise AssertionError("survivor communicator failed after rank removal")

        if rank == 0:
            print(
                json.dumps(
                    {
                        "world_size": world,
                        "gpus": gpu_ids,
                        "hcas": hcas,
                        "transport": args.transport,
                        "qps": group.qps,
                        "sms_and_nccl_ctas": args.sms,
                        "nccl_nvlink_disabled": not args.allow_nvlink_nccl,
                        "nccl_algorithm": os.environ["NCCL_ALGO"],
                        "nccl_protocol": os.environ["NCCL_PROTO"],
                        "precondition_bytes": precondition_bytes,
                        "precondition_iterations": args.precondition_iterations,
                        "dtype_parity": parity,
                        "timeout_recovery_test": timeout_recovery_verified,
                        "shrink_reconfigure_test": shrink_verified,
                        "results": results,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        group.close()
        pd.close()
        context.close()
        nccl_reference.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hcas", required=True, help="comma-separated HCA per rank")
    parser.add_argument("--gpus", type=lambda value: [int(x) for x in value.split(",")])
    parser.add_argument(
        "--sizes", type=_parse_sizes, default=_parse_sizes("1m,16m,64m")
    )
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument(
        "--qps",
        type=int,
        help="QPs per direction (default: one per SM for GPUNetIO, 4 for host)",
    )
    parser.add_argument("--sms", type=int, default=32)
    parser.add_argument("--queue-depth", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--precondition-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--uuid", type=int, default=1)
    parser.add_argument("--reconfigure-count", type=int, default=1)
    parser.add_argument("--advertise-host")
    parser.add_argument("--transport", choices=("gpunetio", "host"), default="gpunetio")
    parser.add_argument("--gpunetio-arch", default="sm_90")
    parser.add_argument("--gpunetio-bitcode")
    parser.add_argument("--nic-handler", choices=("gpu", "auto", "cpu"), default="gpu")
    parser.add_argument(
        "--parity-sizes",
        type=_parse_sizes,
        default=_parse_sizes("16,32752,32768,32784,65520,65536,65552,1m"),
        help="comma-separated exact-byte matrix sizes around scheduler boundaries",
    )
    parser.add_argument(
        "--parity-dtypes",
        type=_parse_dtypes,
        default=list(_DTYPES),
        help="comma-separated exact-byte NCCL parity matrix",
    )
    parser.add_argument("--allow-nvlink-nccl", action="store_true")
    parser.add_argument(
        "--require-bit-exact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--timeout-smoke-test", action="store_true")
    parser.add_argument("--timeout-smoke-seconds", type=float, default=0.05)
    parser.add_argument("--shrink-smoke-test", action="store_true")
    args = parser.parse_args()
    if (
        (args.qps is not None and args.qps <= 0)
        or args.sms <= 0
        or args.queue_depth <= 0
        or args.reconfigure_count <= 0
        or args.precondition_iterations < 0
        or args.timeout_smoke_seconds <= 0
    ):
        parser.error(
            "qps, sms, queue-depth, and reconfigure-count must be positive; "
            "precondition-iterations must be non-negative"
        )
    run(args)


if __name__ == "__main__":
    main()
