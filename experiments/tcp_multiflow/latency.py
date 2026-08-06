#!/usr/bin/env python3
"""Measure physical TCP request/response latency between two local NICs."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from experiments.tcp_multiflow import benchmark

_MODULE = "experiments.tcp_multiflow.latency"


def _size(value: str) -> int:
    suffixes = {"k": 1 << 10, "m": 1 << 20}
    normalized = value.strip().lower()
    multiplier = suffixes.get(normalized[-1], 1)
    if multiplier != 1:
        normalized = normalized[:-1]
    try:
        result = int(normalized) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be positive integers") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("sizes must be positive integers")
    return result


def _sizes(value: str) -> List[int]:
    values = [_size(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one payload size")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("payload sizes must be unique")
    return values


def _label(size: int) -> str:
    if size >= 1 << 20 and size % (1 << 20) == 0:
        return "%d MiB" % (size >> 20)
    if size >= 1 << 10 and size % (1 << 10) == 0:
        return "%d KiB" % (size >> 10)
    return "%d B" % size


def _socket_address(address: str, port: int):
    if ipaddress.ip_address(address).version == 6:
        return (address, port, 0, 0)
    return (address, port)


def _socket_family(address: str):
    return (
        socket.AF_INET6
        if ipaddress.ip_address(address).version == 6
        else socket.AF_INET
    )


def _bind_device(sock: socket.socket, interface: str):
    option = getattr(socket, "SO_BINDTODEVICE", 25)
    sock.setsockopt(socket.SOL_SOCKET, option, interface.encode() + b"\0")


def _receive(sock: socket.socket, size: int, allow_eof: bool = False):
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            if allow_eof and not result:
                return None
            raise RuntimeError("TCP peer closed during a message")
        result.extend(chunk)
    return bytes(result)


def run_server(args) -> int:
    family = _socket_family(args.address)
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _bind_device(listener, args.interface)
        listener.bind(_socket_address(args.address, args.port))
        listener.listen(1)
        connection, _ = listener.accept()
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                payload = _receive(connection, args.size, allow_eof=True)
                if payload is None:
                    return 0
                connection.sendall(payload)


def _percentile(samples: Sequence[int], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index] / 1000


def _counter_delta(before: benchmark.LinkCounters, after: benchmark.LinkCounters):
    return benchmark.LinkCounters(
        bytes=after.bytes - before.bytes,
        packets=after.packets - before.packets,
    )


def _round_trip(sock: socket.socket, payload: bytes):
    sock.sendall(payload)
    response = _receive(sock, len(payload))
    if response != payload:
        raise RuntimeError("TCP response payload did not match request")


def _measure_size(args, sender, receiver, size: int, port: int, cpu_map, root: Path):
    server_command = [
        sys.executable,
        "-m",
        _MODULE,
        "server",
        "--interface",
        receiver.interface,
        "--address",
        receiver.address,
        "--port",
        str(port),
        "--size",
        str(size),
    ]
    log_path = root / ("server-%d.log" % size)
    if args.dry_run:
        print(
            "cpu %s: %s > %s"
            % (cpu_map[receiver.interface][0], " ".join(server_command), log_path)
        )
        return None

    with log_path.open("wb") as output:
        server = subprocess.Popen(
            server_command,
            stdout=output,
            stderr=subprocess.STDOUT,
            preexec_fn=benchmark._affinity_function(cpu_map[receiver.interface][0]),
        )
    try:
        time.sleep(0.25)
        if server.poll() is not None:
            raise RuntimeError("latency server failed to listen (see %s)" % log_path)
        family = _socket_family(sender.address)
        with socket.socket(family, socket.SOCK_STREAM) as client:
            _bind_device(client, sender.interface)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.bind(_socket_address(sender.address, 0))
            client.connect(_socket_address(receiver.address, port))
            payload = bytes((index & 0xFF for index in range(size)))
            for _ in range(args.warmups):
                _round_trip(client, payload)

            sender_tx_before = benchmark._link_counters(sender.interface, "tx")
            sender_rx_before = benchmark._link_counters(sender.interface, "rx")
            receiver_tx_before = benchmark._link_counters(receiver.interface, "tx")
            receiver_rx_before = benchmark._link_counters(receiver.interface, "rx")
            samples = []
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                _round_trip(client, payload)
                samples.append(time.perf_counter_ns() - start)
            sender_tx_after = benchmark._link_counters(sender.interface, "tx")
            sender_rx_after = benchmark._link_counters(sender.interface, "rx")
            receiver_tx_after = benchmark._link_counters(receiver.interface, "tx")
            receiver_rx_after = benchmark._link_counters(receiver.interface, "rx")
        returncode = server.wait(timeout=5)
        if returncode:
            raise RuntimeError(
                "latency server exited %d (see %s)" % (returncode, log_path)
            )
    except BaseException:
        benchmark._stop_processes([server])
        raise

    sender_tx = _counter_delta(sender_tx_before, sender_tx_after)
    sender_rx = _counter_delta(sender_rx_before, sender_rx_after)
    receiver_tx = _counter_delta(receiver_tx_before, receiver_tx_after)
    receiver_rx = _counter_delta(receiver_rx_before, receiver_rx_after)
    request_packets = min(sender_tx.packets, receiver_rx.packets)
    response_packets = min(receiver_tx.packets, sender_rx.packets)
    path_verified = (
        request_packets >= args.iterations and response_packets >= args.iterations
    )
    if not path_verified:
        raise RuntimeError(
            "physical counters did not observe every TCP request and response"
        )
    return {
        "payload_bytes": size,
        "iterations": args.iterations,
        "median_us": statistics.median(samples) / 1000,
        "mean_us": statistics.mean(samples) / 1000,
        "p95_us": _percentile(samples, 0.95),
        "p99_us": _percentile(samples, 0.99),
        "min_us": min(samples) / 1000,
        "max_us": max(samples) / 1000,
        "path_verified": path_verified,
        "request_packets": request_packets,
        "response_packets": response_packets,
        "counter_deltas": {
            "sender_tx": asdict(sender_tx),
            "sender_rx": asdict(sender_rx),
            "receiver_tx": asdict(receiver_tx),
            "receiver_rx": asdict(receiver_rx),
        },
    }


def run(args) -> int:
    if len(args.interfaces) != 2:
        raise RuntimeError("latency requires exactly two interfaces")
    endpoints = benchmark._endpoints(args.interfaces, args.addresses)
    sender, receiver = endpoints
    if args.reverse:
        sender, receiver = receiver, sender
    if (
        ipaddress.ip_address(sender.address).version
        != ipaddress.ip_address(receiver.address).version
    ):
        raise RuntimeError("sender and receiver addresses use different IP versions")
    if args.base_port + len(args.sizes) - 1 > 65535:
        raise RuntimeError("latency port exceeds 65535")
    cpu_map = benchmark._allocate_cpus(endpoints, 1, not args.no_affinity)
    root = args.output_dir or benchmark._default_output_dir("latency")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    original_affinity = None
    if not args.no_affinity and not args.dry_run:
        original_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {cpu_map[sender.interface][0]})
    try:
        for index, size in enumerate(args.sizes):
            result = _measure_size(
                args, sender, receiver, size, args.base_port + index, cpu_map, root
            )
            if result is not None:
                results.append(result)
    finally:
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)
    if args.dry_run:
        return 0
    report = {
        "direction": "%s-to-%s" % (sender.interface, receiver.interface),
        "sender": asdict(sender),
        "receiver": asdict(receiver),
        "cpu_affinity": {
            "sender": cpu_map[sender.interface][0],
            "receiver": cpu_map[receiver.interface][0],
        },
        "warmups": args.warmups,
        "results": results,
    }
    (root / "latency.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("payload  median RTT  p95 RTT  p99 RTT  minimum")
    for result in results:
        print(
            "%7s  %9.2f  %7.2f  %7.2f  %7.2f us"
            % (
                _label(result["payload_bytes"]),
                result["median_us"],
                result["p95_us"],
                result["p99_us"],
                result["min_us"],
            )
        )
    print("Results are in %s" % root)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)

    client = subparsers.add_parser("run", help="run the latency benchmark")
    client.add_argument(
        "--interfaces",
        type=benchmark._csv,
        default=["eth1", "eth2"],
        help="sender and receiver interfaces (default: eth1,eth2)",
    )
    client.add_argument(
        "--addresses",
        type=benchmark._addresses,
        help="sender and receiver addresses (default: auto-detect)",
    )
    client.add_argument(
        "--sizes",
        type=_sizes,
        default=[1, 64, 1 << 10, 4 << 10, 64 << 10],
        help="comma-separated payload bytes with k/m suffixes",
    )
    client.add_argument("--iterations", type=benchmark._positive_int, default=10000)
    client.add_argument("--warmups", type=benchmark._positive_int, default=1000)
    client.add_argument("--base-port", type=benchmark._positive_int, default=5291)
    client.add_argument("--output-dir", type=Path)
    client.add_argument("--reverse", action="store_true")
    client.add_argument("--no-affinity", action="store_true")
    client.add_argument("--dry-run", action="store_true")
    client.set_defaults(function=run)

    server = subparsers.add_parser("server", help="run the internal echo server")
    server.add_argument("--interface", required=True)
    server.add_argument("--address", required=True)
    server.add_argument("--port", type=benchmark._positive_int, required=True)
    server.add_argument("--size", type=benchmark._positive_int, required=True)
    server.set_defaults(function=run_server)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.function(args)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
