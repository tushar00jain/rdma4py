#!/usr/bin/env python3
"""Drive multi-process Python TCP flows over two local physical NICs."""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from experiments.tcp_multiflow import benchmark

_MODULE = "experiments.tcp_multiflow.python_benchmark"
_RECEIVE_BYTES = 1 << 20


def _size(value: str) -> int:
    suffixes = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    normalized = value.strip().lower()
    if not normalized:
        raise argparse.ArgumentTypeError("write size must be positive")
    multiplier = suffixes.get(normalized[-1], 1)
    if multiplier != 1:
        normalized = normalized[:-1]
    try:
        result = int(normalized) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError("write size must be positive") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("write size must be positive")
    return result


def _socket_family(address: str):
    return (
        socket.AF_INET6
        if ipaddress.ip_address(address).version == 6
        else socket.AF_INET
    )


def _socket_address(address: str, port: int):
    if ipaddress.ip_address(address).version == 6:
        return (address, port, 0, 0)
    return (address, port)


def _bind_device(sock: socket.socket, interface: str):
    option = getattr(socket, "SO_BINDTODEVICE", 25)
    sock.setsockopt(socket.SOL_SOCKET, option, interface.encode() + b"\0")


def _total_retransmits(sock: socket.socket) -> int:
    option = getattr(socket, "TCP_INFO", None)
    if option is None:
        return 0
    info = sock.getsockopt(socket.IPPROTO_TCP, option, 104)
    if len(info) < 104:
        return 0
    return struct.unpack_from("=I", info, 100)[0]


def run_server(args) -> int:
    """Receive one internal worker flow until its sender closes."""

    family = _socket_family(args.address)
    buffer = bytearray(_RECEIVE_BYTES)
    view = memoryview(buffer)
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _bind_device(listener, args.interface)
        listener.bind(_socket_address(args.address, args.port))
        listener.listen(1)
        connection, _ = listener.accept()
        with connection:
            total = 0
            start_ns = time.perf_counter_ns()
            while True:
                received = connection.recv_into(view)
                if not received:
                    break
                total += received
            elapsed_seconds = (time.perf_counter_ns() - start_ns) / 1e9
    print(
        json.dumps(
            {"bytes_received": total, "elapsed_seconds": elapsed_seconds},
            sort_keys=True,
        )
    )
    return 0


def run_client(args) -> int:
    """Send one internal worker flow and emit its measured JSON result."""

    family = _socket_family(args.local_address)
    payload = bytes(args.length)
    with socket.socket(family, socket.SOCK_STREAM) as client:
        _bind_device(client, args.interface)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client.bind(_socket_address(args.local_address, 0))
        client.connect(_socket_address(args.server_address, args.port))

        start_ns = time.perf_counter_ns()
        measured_start_ns = start_ns + int(args.omit * 1e9)
        deadline_ns = measured_start_ns + int(args.duration * 1e9)
        measured_bytes = 0
        while True:
            client.sendall(payload)
            now_ns = time.perf_counter_ns()
            if now_ns >= measured_start_ns:
                measured_bytes += len(payload)
            if now_ns >= deadline_ns:
                break
        retransmits = _total_retransmits(client)
        measured_seconds = (now_ns - measured_start_ns) / 1e9
    print(
        json.dumps(
            {
                "bytes_sent": measured_bytes,
                "elapsed_seconds": measured_seconds,
                "retransmits": retransmits,
            },
            sort_keys=True,
        )
    )
    return 0


def _worker_specs(sender, receiver, args, cpu_map):
    servers = []
    clients = []
    for flow in range(args.flows_per_nic):
        port = args.base_port + flow
        servers.append(
            benchmark.ProcessSpec(
                interface=receiver.interface,
                flow=flow,
                port=port,
                cpu=cpu_map[receiver.interface][flow],
                command=(
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
                ),
            )
        )
        clients.append(
            benchmark.ProcessSpec(
                interface=sender.interface,
                flow=flow,
                port=port,
                cpu=cpu_map[sender.interface][flow],
                command=(
                    sys.executable,
                    "-m",
                    _MODULE,
                    "client",
                    "--interface",
                    sender.interface,
                    "--local-address",
                    sender.address,
                    "--server-address",
                    receiver.address,
                    "--port",
                    str(port),
                    "--length",
                    str(args.length),
                    "--duration",
                    str(args.duration),
                    "--omit",
                    str(args.omit),
                ),
            )
        )
    return servers, clients


def _load_result(path: Path):
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Python worker produced invalid JSON in %s" % path
        ) from error
    if "bytes_sent" not in result or "elapsed_seconds" not in result:
        raise RuntimeError("Python client result is incomplete in %s" % path)
    return result


def _flow_results(specs, paths):
    results = []
    for spec, path in zip(specs, paths):
        document = _load_result(path)
        seconds = float(document["elapsed_seconds"])
        if seconds <= 0:
            raise RuntimeError("Python client measured no time in %s" % path)
        results.append(
            {
                "interface": spec.interface,
                "flow": spec.flow,
                "port": spec.port,
                "cpu": spec.cpu,
                "gbps": int(document["bytes_sent"]) * 8 / seconds / 1e9,
                "retransmits": int(document.get("retransmits", 0)),
                "bytes_sent": int(document["bytes_sent"]),
                "elapsed_seconds": seconds,
            }
        )
    return results


def run_local(args) -> int:
    """Run Python TCP processes across two local, physically connected NICs."""

    if len(args.interfaces) != 2:
        raise RuntimeError("this experiment requires exactly two interfaces")
    if args.base_port + args.flows_per_nic - 1 > 65535:
        raise RuntimeError("flow port exceeds 65535")
    if args.counter_delay + args.sample_seconds >= args.duration:
        raise RuntimeError(
            "counter delay plus sample seconds must be less than measured duration"
        )
    endpoints = benchmark._endpoints(args.interfaces, args.addresses)
    sender, receiver = endpoints
    if args.reverse:
        sender, receiver = receiver, sender
    if (
        ipaddress.ip_address(sender.address).version
        != ipaddress.ip_address(receiver.address).version
    ):
        raise RuntimeError("sender and receiver addresses use different IP versions")
    if sender.speed_gbps is not None and sender.speed_gbps < args.target_gbps:
        raise RuntimeError(
            "%s is a %.1f Gb/s link, below the %.1f Gb/s target"
            % (sender.interface, sender.speed_gbps, args.target_gbps)
        )

    cpu_map = benchmark._allocate_cpus(
        endpoints, args.flows_per_nic, not args.no_affinity
    )
    server_specs, client_specs = _worker_specs(sender, receiver, args, cpu_map)
    output_dir = args.output_dir or benchmark._default_output_dir("python-local")
    server_paths = benchmark._process_paths(server_specs, output_dir / "server")
    client_paths = benchmark._process_paths(client_specs, output_dir / "client")
    benchmark._write_metadata(
        output_dir,
        "python-local",
        endpoints,
        client_specs,
        {
            "engine": "python",
            "python_version": sys.version,
            "server_flows": [asdict(spec) for spec in server_specs],
            "write_size": args.length,
            "target_gbps": args.target_gbps,
            "counter_delay": args.counter_delay,
            "sample_seconds": args.sample_seconds,
            "direction": "reverse" if args.reverse else "forward",
        },
    )
    if args.dry_run:
        print("server processes:")
        benchmark._print_processes(server_specs, server_paths)
        print("client processes:")
        benchmark._print_processes(client_specs, client_paths)
        return 0

    server_processes = benchmark._launch_processes(server_specs, server_paths)
    client_processes = []
    try:
        time.sleep(0.5)
        for spec, process, path in zip(server_specs, server_processes, server_paths):
            if process.poll() is not None:
                raise RuntimeError(
                    "%s flow %d failed to listen (see %s)"
                    % (spec.interface, spec.flow, path)
                )
        client_processes = benchmark._launch_processes(client_specs, client_paths)
        time.sleep(args.omit + args.counter_delay)
        for spec, process, path in zip(client_specs, client_processes, client_paths):
            if process.poll() is not None:
                raise RuntimeError(
                    "%s flow %d ended before counter sampling (see %s)"
                    % (spec.interface, spec.flow, path)
                )

        tx_before = benchmark._link_counters(sender.interface, "tx")
        rx_before = benchmark._link_counters(receiver.interface, "rx")
        sample_start = time.perf_counter_ns()
        time.sleep(args.sample_seconds)
        sample_seconds = (time.perf_counter_ns() - sample_start) / 1e9
        tx_after = benchmark._link_counters(sender.interface, "tx")
        rx_after = benchmark._link_counters(receiver.interface, "rx")

        benchmark._wait_processes(client_specs, client_processes, client_paths)
        benchmark._wait_processes(server_specs, server_processes, server_paths)
    except BaseException:
        benchmark._stop_processes(client_processes)
        benchmark._stop_processes(server_processes)
        raise

    flows = _flow_results(client_specs, client_paths)
    payload_gbps = sum(flow["gbps"] for flow in flows)
    tx_wire_gbps = benchmark._wire_gbps(tx_before, tx_after, sample_seconds)
    rx_wire_gbps = benchmark._wire_gbps(rx_before, rx_after, sample_seconds)
    wire_gbps = min(tx_wire_gbps, rx_wire_gbps)
    counter_difference = abs(tx_wire_gbps - rx_wire_gbps) / max(
        tx_wire_gbps, rx_wire_gbps, 1.0
    )
    path_verified = wire_gbps > 0 and counter_difference <= 0.02
    passed = path_verified and wire_gbps >= args.target_gbps
    utilization = wire_gbps / sender.speed_gbps * 100 if sender.speed_gbps else None
    summary = {
        "engine": "python",
        "passed": passed,
        "path_verified": path_verified,
        "direction": "reverse" if args.reverse else "forward",
        "write_size": args.length,
        "flows": flows,
        "flow_count": len(flows),
        "payload_gbps": payload_gbps,
        "wire_gbps": wire_gbps,
        "tx_wire_gbps": tx_wire_gbps,
        "rx_wire_gbps": rx_wire_gbps,
        "link_speed_gbps": sender.speed_gbps,
        "link_utilization_percent": utilization,
        "target_gbps": args.target_gbps,
        "sample_seconds": sample_seconds,
        "total_retransmits": sum(flow["retransmits"] for flow in flows),
        "counter_samples": {
            "tx_before": asdict(tx_before),
            "tx_after": asdict(tx_after),
            "rx_before": asdict(rx_before),
            "rx_after": asdict(rx_after),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        "Python TCP payload: %.2f Gb/s across %d flows (%d retransmits)"
        % (payload_gbps, len(flows), summary["total_retransmits"])
    )
    print(
        "%s TX / %s RX wire rate: %.2f / %.2f Gb/s"
        % (sender.interface, receiver.interface, tx_wire_gbps, rx_wire_gbps)
    )
    if utilization is not None:
        print(
            "line rate: %.2f Gb/s (%.2f%% of %.0f Gb/s): %s"
            % (
                wire_gbps,
                utilization,
                sender.speed_gbps,
                "PASS" if passed else "FAIL",
            )
        )
    else:
        print(
            "line rate: %.2f Gb/s (target %.2f Gb/s): %s"
            % (wire_gbps, args.target_gbps, "PASS" if passed else "FAIL")
        )
    print("Results are in %s" % output_dir)
    return 0 if passed else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)

    local = subparsers.add_parser("local", help="run local Python TCP flows")
    local.add_argument(
        "--interfaces",
        type=benchmark._csv,
        default=["eth1", "eth2"],
        help="sender and receiver interfaces (default: eth1,eth2)",
    )
    local.add_argument(
        "--addresses",
        type=benchmark._addresses,
        help="sender and receiver addresses (default: auto-detect)",
    )
    local.add_argument(
        "--flows-per-nic",
        type=benchmark._positive_int,
        default=24,
        help="independent Python sender processes (default: 24)",
    )
    local.add_argument(
        "--base-port",
        type=benchmark._positive_int,
        default=5201,
        help="first TCP flow port (default: 5201)",
    )
    local.add_argument(
        "--duration",
        type=benchmark._positive_int,
        default=15,
        help="measured seconds after warmup (default: 15)",
    )
    local.add_argument(
        "--omit",
        type=benchmark._nonnegative_int,
        default=2,
        help="warmup seconds (default: 2)",
    )
    local.add_argument(
        "--counter-delay",
        type=benchmark._nonnegative_int,
        default=7,
        help="measured seconds before counter sampling (default: 7)",
    )
    local.add_argument(
        "--sample-seconds",
        type=benchmark._positive_int,
        default=5,
        help="steady-state physical-counter sample (default: 5)",
    )
    local.add_argument(
        "--length",
        type=_size,
        default=1 << 20,
        help="bytes per socket.sendall call with K/M/G suffixes (default: 1M)",
    )
    local.add_argument(
        "--target-gbps",
        type=benchmark._nonnegative_float,
        default=195.0,
        help="physical wire-rate pass threshold (default: 195)",
    )
    local.add_argument(
        "--output-dir", type=Path, help="directory for JSON results and metadata"
    )
    local.add_argument(
        "--reverse", action="store_true", help="swap sender and receiver interfaces"
    )
    local.add_argument(
        "--no-affinity", action="store_true", help="do not pin NIC-local workers"
    )
    local.add_argument(
        "--dry-run", action="store_true", help="print worker commands only"
    )
    local.set_defaults(function=run_local)

    server = subparsers.add_parser("server", help="run an internal TCP receiver")
    server.add_argument("--interface", required=True)
    server.add_argument("--address", required=True)
    server.add_argument("--port", type=benchmark._positive_int, required=True)
    server.set_defaults(function=run_server)

    client = subparsers.add_parser("client", help="run an internal TCP sender")
    client.add_argument("--interface", required=True)
    client.add_argument("--local-address", required=True)
    client.add_argument("--server-address", required=True)
    client.add_argument("--port", type=benchmark._positive_int, required=True)
    client.add_argument("--length", type=_size, required=True)
    client.add_argument("--duration", type=benchmark._positive_int, required=True)
    client.add_argument("--omit", type=benchmark._nonnegative_int, required=True)
    client.set_defaults(function=run_client)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments and run the requested benchmark role."""

    args = _parser().parse_args(argv)
    try:
        return args.function(args)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
