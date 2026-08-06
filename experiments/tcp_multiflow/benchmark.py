#!/usr/bin/env python3
"""Drive enough independent iperf3 processes to fill two fast Ethernet NICs.

Each TCP flow gets its own iperf3 process.  This is intentional: iperf3 releases
before 3.16 schedule all parallel streams in one thread, which can make CPU
throughput look like a network limit on 100 Gb/s and faster links.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Endpoint:
    """A local benchmark interface and the IP address bound on it."""

    interface: str
    address: str
    speed_gbps: Optional[float]
    numa_node: Optional[int]


@dataclass(frozen=True)
class ProcessSpec:
    """One independently scheduled TCP flow."""

    interface: str
    flow: int
    port: int
    cpu: Optional[int]
    command: Tuple[str, ...]


@dataclass(frozen=True)
class FlowResult:
    """The receiver goodput and retransmits reported for one TCP flow."""

    interface: str
    flow: int
    port: int
    cpu: Optional[int]
    gbps: float
    retransmits: int
    bytes_received: int


@dataclass(frozen=True)
class LinkCounters:
    """Physical-layer octets and packets sampled from a NIC."""

    bytes: int
    packets: int


def _csv(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("list entries must be unique")
    return values


def _addresses(value: str) -> List[str]:
    values = _csv(value)
    try:
        values = [str(ipaddress.ip_address(address)) for address in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError("addresses must be IP addresses") from error
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("addresses must be unique")
    return values


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return result


def _run_json(command: Sequence[str]):
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("%s is required" % command[0]) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError("%s failed: %s" % (shlex.join(command), detail)) from error
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("%s returned invalid JSON" % command[0]) from error


def _interface_ip_addresses(interface: str) -> List[str]:
    records = _run_json(["ip", "-json", "address", "show", "dev", interface])
    if not records:
        raise RuntimeError("interface %s does not exist" % interface)
    addresses = []
    for record in records:
        for info in record.get("addr_info", []):
            if (
                info.get("family") in {"inet", "inet6"}
                and info.get("scope") == "global"
            ):
                addresses.append(str(ipaddress.ip_address(info["local"])))
    if not addresses:
        raise RuntimeError("interface %s has no global IP address" % interface)
    return addresses


def _read_optional(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, OSError):
        return None


def _interface_speed(interface: str) -> Optional[float]:
    value = _read_optional(Path("/sys/class/net") / interface / "speed")
    try:
        speed_mbps = int(value) if value is not None else -1
    except ValueError:
        speed_mbps = -1
    return speed_mbps / 1000 if speed_mbps > 0 else None


def _interface_numa_node(interface: str) -> Optional[int]:
    value = _read_optional(Path("/sys/class/net") / interface / "device/numa_node")
    try:
        node = int(value) if value is not None else -1
    except ValueError:
        node = -1
    return node if node >= 0 else None


def _check_carrier(interface: str):
    root = Path("/sys/class/net") / interface
    carrier = _read_optional(root / "carrier")
    state = _read_optional(root / "operstate")
    if carrier == "0" or state == "down":
        raise RuntimeError("interface %s has no carrier" % interface)


def _endpoints(
    interfaces: Sequence[str], requested_addresses: Optional[Sequence[str]]
) -> List[Endpoint]:
    if requested_addresses is not None and len(requested_addresses) != len(interfaces):
        raise RuntimeError("provide exactly one address per interface")
    endpoints = []
    for index, interface in enumerate(interfaces):
        available = _interface_ip_addresses(interface)
        address = (
            requested_addresses[index]
            if requested_addresses is not None
            else available[0]
        )
        if address not in available:
            raise RuntimeError("%s is not assigned to %s" % (address, interface))
        _check_carrier(interface)
        endpoints.append(
            Endpoint(
                interface=interface,
                address=address,
                speed_gbps=_interface_speed(interface),
                numa_node=_interface_numa_node(interface),
            )
        )
    return endpoints


def _parse_cpu_list(value: str) -> List[int]:
    cpus = []
    for item in value.split(","):
        bounds = item.split("-", 1)
        first = int(bounds[0])
        last = int(bounds[-1])
        if last < first:
            raise ValueError("invalid CPU range %s" % item)
        cpus.extend(range(first, last + 1))
    return cpus


def _cpu_pool(endpoint: Endpoint) -> List[int]:
    allowed = set(os.sched_getaffinity(0))
    if endpoint.numa_node is not None:
        cpulist = _read_optional(
            Path("/sys/devices/system/node")
            / ("node%d" % endpoint.numa_node)
            / "cpulist"
        )
        if cpulist:
            local = sorted(allowed.intersection(_parse_cpu_list(cpulist)))
            if local:
                return local
    return sorted(allowed)


def _allocate_cpus(
    endpoints: Sequence[Endpoint], flows_per_nic: int, affinity: bool
) -> Dict[str, List[Optional[int]]]:
    if not affinity:
        return {endpoint.interface: [None] * flows_per_nic for endpoint in endpoints}
    pools = {endpoint.interface: _cpu_pool(endpoint) for endpoint in endpoints}
    used = set()
    result = {}
    for endpoint in endpoints:
        pool = pools[endpoint.interface]
        unused = [cpu for cpu in pool if cpu not in used]
        candidates = unused + [cpu for cpu in pool if cpu in used]
        if not candidates:
            raise RuntimeError("no CPUs are available to run iperf3")
        assigned = [
            candidates[index % len(candidates)] for index in range(flows_per_nic)
        ]
        used.update(assigned)
        result[endpoint.interface] = assigned
    return result


def _check_routes(endpoints: Sequence[Endpoint], server_addresses: Sequence[str]):
    for endpoint, server_address in zip(endpoints, server_addresses):
        version = ipaddress.ip_address(server_address).version
        records = _run_json(
            [
                "ip",
                "-json",
                "-6" if version == 6 else "-4",
                "route",
                "get",
                server_address,
                "from",
                endpoint.address,
            ]
        )
        route = records[0] if records else {}
        if route.get("dev") != endpoint.interface:
            raise RuntimeError(
                "route from %s to %s uses %s, not %s"
                % (
                    endpoint.address,
                    server_address,
                    route.get("dev", "no interface"),
                    endpoint.interface,
                )
            )


def _check_iperf3(executable: str):
    path = shutil.which(executable)
    if path is None:
        raise RuntimeError("iperf3 is required but %s was not found" % executable)


def _server_specs(
    endpoints: Sequence[Endpoint], args, cpus: Dict[str, List[Optional[int]]]
) -> List[ProcessSpec]:
    specs = []
    for endpoint in endpoints:
        version = ipaddress.ip_address(endpoint.address).version
        for flow in range(args.flows_per_nic):
            port = args.base_port + flow
            specs.append(
                ProcessSpec(
                    interface=endpoint.interface,
                    flow=flow,
                    port=port,
                    cpu=cpus[endpoint.interface][flow],
                    command=(
                        args.iperf3,
                        "--version6" if version == 6 else "--version4",
                        "--server",
                        "--one-off",
                        "--bind",
                        endpoint.address,
                        "--bind-dev",
                        endpoint.interface,
                        "--port",
                        str(port),
                        "--json",
                    ),
                )
            )
    return specs


def _client_specs(
    endpoints: Sequence[Endpoint], args, cpus: Dict[str, List[Optional[int]]]
) -> List[ProcessSpec]:
    specs = []
    for endpoint, server_address in zip(endpoints, args.server_addresses):
        version = ipaddress.ip_address(server_address).version
        if ipaddress.ip_address(endpoint.address).version != version:
            raise RuntimeError(
                "%s and %s use different IP versions"
                % (endpoint.address, server_address)
            )
        for flow in range(args.flows_per_nic):
            port = args.base_port + flow
            command = [
                args.iperf3,
                "--version6" if version == 6 else "--version4",
                "--client",
                server_address,
                "--bind",
                endpoint.address,
                "--bind-dev",
                endpoint.interface,
                "--port",
                str(port),
                "--time",
                str(args.duration),
                "--omit",
                str(args.omit),
                "--length",
                args.length,
                "--zerocopy",
                "--json",
            ]
            if args.reverse:
                command.append("--reverse")
            specs.append(
                ProcessSpec(
                    interface=endpoint.interface,
                    flow=flow,
                    port=port,
                    cpu=cpus[endpoint.interface][flow],
                    command=tuple(command),
                )
            )
    return specs


def _affinity_function(cpu: Optional[int]):
    if cpu is None:
        return None

    def set_affinity():
        os.sched_setaffinity(0, {cpu})

    return set_affinity


def _stop_processes(processes: Sequence[subprocess.Popen]):
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


def _process_paths(specs: Sequence[ProcessSpec], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        output_dir / ("%s-flow-%02d.json" % (spec.interface, spec.flow))
        for spec in specs
    ]


def _print_processes(specs: Sequence[ProcessSpec], paths: Sequence[Path]):
    for spec, path in zip(specs, paths):
        prefix = "cpu %s: " % spec.cpu if spec.cpu is not None else ""
        print("%s%s > %s" % (prefix, shlex.join(spec.command), path))


def _launch_processes(
    specs: Sequence[ProcessSpec], paths: Sequence[Path]
) -> List[subprocess.Popen]:
    processes = []
    try:
        for spec, path in zip(specs, paths):
            with path.open("wb") as output:
                processes.append(
                    subprocess.Popen(
                        spec.command,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        preexec_fn=_affinity_function(spec.cpu),
                    )
                )
    except BaseException:
        _stop_processes(processes)
        raise
    return processes


def _wait_processes(
    specs: Sequence[ProcessSpec],
    processes: Sequence[subprocess.Popen],
    paths: Sequence[Path],
):
    failures = []
    for spec, process, path in zip(specs, processes, paths):
        returncode = process.wait()
        if returncode:
            failures.append(
                "%s flow %d exited %d (see %s)"
                % (spec.interface, spec.flow, returncode, path)
            )
    if failures:
        raise RuntimeError("; ".join(failures))


def _run_processes(
    specs: Sequence[ProcessSpec], output_dir: Path, dry_run: bool
) -> List[Path]:
    paths = _process_paths(specs, output_dir)
    if dry_run:
        _print_processes(specs, paths)
        return paths
    processes = _launch_processes(specs, paths)
    try:
        _wait_processes(specs, processes, paths)
    except BaseException:
        _stop_processes(processes)
        raise
    return paths


def _load_document(path: Path):
    text = path.read_text(errors="replace")
    start = text.find("{")
    if start < 0:
        raise RuntimeError("iperf3 produced no JSON in %s" % path)
    try:
        document, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise RuntimeError("iperf3 produced invalid JSON in %s" % path) from error
    if document.get("error"):
        raise RuntimeError("iperf3 failed for %s: %s" % (path, document["error"]))
    return document


def _flow_result(spec: ProcessSpec, document) -> FlowResult:
    end = document.get("end", {})
    received = end.get("sum_received") or end.get("sum_sent")
    sent = end.get("sum_sent", {})
    if not received or "bits_per_second" not in received:
        raise RuntimeError(
            "iperf3 result for %s flow %d has no throughput"
            % (spec.interface, spec.flow)
        )
    return FlowResult(
        interface=spec.interface,
        flow=spec.flow,
        port=spec.port,
        cpu=spec.cpu,
        gbps=float(received["bits_per_second"]) / 1e9,
        retransmits=int(sent.get("retransmits", 0)),
        bytes_received=int(received.get("bytes", 0)),
    )


def _link_counters(interface: str, direction: str) -> LinkCounters:
    if direction not in {"rx", "tx"}:
        raise ValueError("counter direction must be rx or tx")
    try:
        completed = subprocess.run(
            ["ethtool", "-S", interface],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ethtool is required for physical counters") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "cannot read physical counters for %s: %s"
            % (interface, error.stderr.strip())
        ) from error
    stats = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split(":", 1)
        if len(fields) == 2:
            try:
                stats[fields[0]] = int(fields[1].strip())
            except ValueError:
                pass
    bytes_name = "%s_bytes_phy" % direction
    packets_name = "%s_packets_phy" % direction
    if bytes_name not in stats or packets_name not in stats:
        raise RuntimeError(
            "%s does not expose %s/%s hardware counters"
            % (interface, bytes_name, packets_name)
        )
    return LinkCounters(stats[bytes_name], stats[packets_name])


def _wire_gbps(before: LinkCounters, after: LinkCounters, seconds: float) -> float:
    byte_delta = after.bytes - before.bytes
    packet_delta = after.packets - before.packets
    if seconds <= 0 or byte_delta < 0 or packet_delta < 0:
        raise RuntimeError("invalid physical counter sample")
    # mlx5 *_bytes_phy includes the frame check sequence. Add the 8-byte
    # preamble/SFD and 12-byte inter-packet gap to get serialized line usage.
    wire_bytes = byte_delta + 20 * packet_delta
    return wire_bytes * 8 / seconds / 1e9


def _summarize(
    endpoints: Sequence[Endpoint], results: Sequence[FlowResult]
) -> Dict[str, Any]:
    nics = {}
    for endpoint in endpoints:
        selected = [
            result for result in results if result.interface == endpoint.interface
        ]
        nics[endpoint.interface] = {
            "address": endpoint.address,
            "link_speed_gbps": endpoint.speed_gbps,
            "flows": len(selected),
            "gbps": sum(result.gbps for result in selected),
            "retransmits": sum(result.retransmits for result in selected),
            "bytes_received": sum(result.bytes_received for result in selected),
        }
    return {
        "nics": nics,
        "total_gbps": sum(result.gbps for result in results),
        "total_retransmits": sum(result.retransmits for result in results),
    }


def _write_metadata(
    output_dir: Path,
    role: str,
    endpoints: Sequence[Endpoint],
    specs: Sequence[ProcessSpec],
    extra: Optional[Dict[str, object]] = None,
):
    metadata = {
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": [asdict(endpoint) for endpoint in endpoints],
        "flows": [asdict(spec) for spec in specs],
    }
    if extra:
        metadata.update(extra)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _default_output_dir(role: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("tcp_line_rate_results") / ("%s-%s" % (role, stamp))


def _validate_common(args):
    if len(args.interfaces) != 2:
        raise RuntimeError("this experiment requires exactly two interfaces")
    if args.base_port + args.flows_per_nic - 1 > 65535:
        raise RuntimeError("flow port exceeds 65535")
    if not args.dry_run:
        _check_iperf3(args.iperf3)


def run_server(args) -> int:
    _validate_common(args)
    endpoints = _endpoints(args.interfaces, args.addresses)
    cpus = _allocate_cpus(endpoints, args.flows_per_nic, not args.no_affinity)
    specs = _server_specs(endpoints, args, cpus)
    output_dir = args.output_dir or _default_output_dir("server")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(output_dir, "server", endpoints, specs)
    print(
        "Listening with %d flows on each NIC; start the client now."
        % args.flows_per_nic
    )
    _run_processes(specs, output_dir, args.dry_run)
    if not args.dry_run:
        print("All server flows completed; raw results are in %s" % output_dir)
    return 0


def _check_link_capacity(endpoints: Sequence[Endpoint], args):
    known = [endpoint.speed_gbps for endpoint in endpoints if endpoint.speed_gbps]
    if len(known) == len(endpoints) and sum(known) < args.target_gbps:
        raise RuntimeError(
            "link capacity %.1f Gb/s is below the %.1f Gb/s target"
            % (sum(known), args.target_gbps)
        )
    for endpoint in endpoints:
        if (
            endpoint.speed_gbps is not None
            and endpoint.speed_gbps < args.minimum_nic_gbps
        ):
            raise RuntimeError(
                "%s is a %.1f Gb/s link, below its %.1f Gb/s target"
                % (
                    endpoint.interface,
                    endpoint.speed_gbps,
                    args.minimum_nic_gbps,
                )
            )


def run_client(args) -> int:
    _validate_common(args)
    if len(args.server_addresses) != len(args.interfaces):
        raise RuntimeError("provide exactly one server address per interface")
    endpoints = _endpoints(args.interfaces, args.local_addresses)
    _check_link_capacity(endpoints, args)
    if not args.skip_route_check:
        _check_routes(endpoints, args.server_addresses)
    cpus = _allocate_cpus(endpoints, args.flows_per_nic, not args.no_affinity)
    specs = _client_specs(endpoints, args, cpus)
    output_dir = args.output_dir or _default_output_dir("client")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        output_dir,
        "client",
        endpoints,
        specs,
        {
            "server_addresses": args.server_addresses,
            "target_gbps": args.target_gbps,
            "minimum_nic_gbps": args.minimum_nic_gbps,
        },
    )
    paths = _run_processes(specs, output_dir, args.dry_run)
    if args.dry_run:
        return 0

    results = [
        _flow_result(spec, _load_document(path)) for spec, path in zip(specs, paths)
    ]
    summary = _summarize(endpoints, results)
    summary["target_gbps"] = args.target_gbps
    summary["minimum_nic_gbps"] = args.minimum_nic_gbps
    summary["flows"] = [asdict(result) for result in results]
    nic_passes = all(
        nic["gbps"] >= args.minimum_nic_gbps for nic in summary["nics"].values()
    )
    passed = summary["total_gbps"] >= args.target_gbps and nic_passes
    summary["passed"] = passed
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    for interface, nic in summary["nics"].items():
        print(
            "%s: %7.2f Gb/s, %2d flows, %d retransmits"
            % (interface, nic["gbps"], nic["flows"], nic["retransmits"])
        )
    print(
        "total: %6.2f Gb/s (target %.2f Gb/s): %s"
        % (
            summary["total_gbps"],
            args.target_gbps,
            "PASS" if passed else "FAIL",
        )
    )
    print("Results are in %s" % output_dir)
    return 0 if passed else 2


def run_local(args) -> int:
    """Run a physical eth1-to-eth2 test on one host without privileges."""

    _validate_common(args)
    if args.counter_delay + args.sample_seconds >= args.duration:
        raise RuntimeError(
            "counter delay plus sample seconds must be less than measured duration"
        )
    endpoints = _endpoints(args.interfaces, args.addresses)
    sender, receiver = endpoints
    if (
        ipaddress.ip_address(sender.address).version
        != ipaddress.ip_address(receiver.address).version
    ):
        raise RuntimeError("sender and receiver addresses use different IP versions")
    for endpoint in endpoints:
        if endpoint.speed_gbps is not None and endpoint.speed_gbps < args.target_gbps:
            raise RuntimeError(
                "%s is a %.1f Gb/s link, below the %.1f Gb/s target"
                % (endpoint.interface, endpoint.speed_gbps, args.target_gbps)
            )

    cpus = _allocate_cpus(endpoints, args.flows_per_nic, not args.no_affinity)
    local_args = argparse.Namespace(**vars(args))
    local_args.server_addresses = [receiver.address]
    server_specs = _server_specs([receiver], local_args, cpus)
    client_specs = _client_specs([sender], local_args, cpus)
    output_dir = args.output_dir or _default_output_dir("local")
    server_paths = _process_paths(server_specs, output_dir / "server")
    client_paths = _process_paths(client_specs, output_dir / "client")
    _write_metadata(
        output_dir,
        "local",
        endpoints,
        client_specs,
        {
            "server_flows": [asdict(spec) for spec in server_specs],
            "target_gbps": args.target_gbps,
            "counter_delay": args.counter_delay,
            "sample_seconds": args.sample_seconds,
            "direction": "reverse" if args.reverse else "forward",
        },
    )
    if args.dry_run:
        print("server processes:")
        _print_processes(server_specs, server_paths)
        print("client processes:")
        _print_processes(client_specs, client_paths)
        return 0

    server_processes = _launch_processes(server_specs, server_paths)
    client_processes = []
    try:
        time.sleep(0.5)
        for spec, process, path in zip(server_specs, server_processes, server_paths):
            if process.poll() is not None:
                raise RuntimeError(
                    "%s flow %d failed to listen (see %s)"
                    % (spec.interface, spec.flow, path)
                )
        client_processes = _launch_processes(client_specs, client_paths)
        time.sleep(args.omit + args.counter_delay)
        for spec, process, path in zip(client_specs, client_processes, client_paths):
            if process.poll() is not None:
                raise RuntimeError(
                    "%s flow %d ended before counter sampling (see %s)"
                    % (spec.interface, spec.flow, path)
                )

        data_sender = receiver if args.reverse else sender
        data_receiver = sender if args.reverse else receiver
        tx_before = _link_counters(data_sender.interface, "tx")
        rx_before = _link_counters(data_receiver.interface, "rx")
        sample_start = time.perf_counter_ns()
        time.sleep(args.sample_seconds)
        sample_seconds = (time.perf_counter_ns() - sample_start) / 1e9
        tx_after = _link_counters(data_sender.interface, "tx")
        rx_after = _link_counters(data_receiver.interface, "rx")

        _wait_processes(client_specs, client_processes, client_paths)
        _wait_processes(server_specs, server_processes, server_paths)
    except BaseException:
        _stop_processes(client_processes)
        _stop_processes(server_processes)
        raise

    results = [
        _flow_result(spec, _load_document(path))
        for spec, path in zip(client_specs, client_paths)
    ]
    payload_gbps = sum(result.gbps for result in results)
    tx_wire_gbps = _wire_gbps(tx_before, tx_after, sample_seconds)
    rx_wire_gbps = _wire_gbps(rx_before, rx_after, sample_seconds)
    wire_gbps = min(tx_wire_gbps, rx_wire_gbps)
    counter_difference = abs(tx_wire_gbps - rx_wire_gbps) / max(
        tx_wire_gbps, rx_wire_gbps, 1.0
    )
    path_verified = wire_gbps > 0 and counter_difference <= 0.02
    passed = path_verified and wire_gbps >= args.target_gbps
    link_speed = data_sender.speed_gbps
    utilization = wire_gbps / link_speed * 100 if link_speed else None
    summary = {
        "passed": passed,
        "path_verified": path_verified,
        "direction": "reverse" if args.reverse else "forward",
        "flows": [asdict(result) for result in results],
        "flow_count": len(results),
        "payload_gbps": payload_gbps,
        "wire_gbps": wire_gbps,
        "tx_wire_gbps": tx_wire_gbps,
        "rx_wire_gbps": rx_wire_gbps,
        "link_speed_gbps": link_speed,
        "link_utilization_percent": utilization,
        "target_gbps": args.target_gbps,
        "sample_seconds": sample_seconds,
        "total_retransmits": sum(result.retransmits for result in results),
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
        "TCP payload: %.2f Gb/s across %d flows (%d retransmits)"
        % (payload_gbps, len(results), summary["total_retransmits"])
    )
    print(
        "%s TX / %s RX wire rate: %.2f / %.2f Gb/s"
        % (
            data_sender.interface,
            data_receiver.interface,
            tx_wire_gbps,
            rx_wire_gbps,
        )
    )
    if utilization is not None:
        print(
            "line rate: %.2f Gb/s (%.2f%% of %.0f Gb/s): %s"
            % (
                wire_gbps,
                utilization,
                link_speed,
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


def _add_common(parser: argparse.ArgumentParser, flows_per_nic: int = 8):
    parser.add_argument(
        "--interfaces",
        type=_csv,
        default=["eth1", "eth2"],
        help="two local interfaces (default: eth1,eth2)",
    )
    parser.add_argument(
        "--flows-per-nic",
        type=_positive_int,
        default=flows_per_nic,
        help="independent iperf3 processes per NIC (default: %d)" % flows_per_nic,
    )
    parser.add_argument(
        "--base-port",
        type=_positive_int,
        default=5201,
        help="first server port on each address (default: 5201)",
    )
    parser.add_argument("--iperf3", default="iperf3", help="iperf3 executable")
    parser.add_argument(
        "--output-dir", type=Path, help="directory for JSON results and metadata"
    )
    parser.add_argument(
        "--no-affinity", action="store_true", help="do not pin flows to NIC-local CPUs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running iperf3"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)

    local = subparsers.add_parser(
        "local", help="benchmark the physical path from eth1 to eth2 on this host"
    )
    _add_common(local, flows_per_nic=24)
    local.add_argument(
        "--addresses",
        type=_addresses,
        help="sender and receiver IP addresses (default: auto-detect)",
    )
    local.add_argument(
        "--duration",
        type=_positive_int,
        default=15,
        help="measured iperf3 seconds (default: 15)",
    )
    local.add_argument(
        "--omit", type=_nonnegative_int, default=2, help="warmup seconds (default: 2)"
    )
    local.add_argument(
        "--counter-delay",
        type=_nonnegative_int,
        default=7,
        help="measured seconds to settle before counter sampling (default: 7)",
    )
    local.add_argument(
        "--sample-seconds",
        type=_positive_int,
        default=5,
        help="steady-state hardware-counter sample (default: 5)",
    )
    local.add_argument("--length", default="1M", help="iperf3 write size (default: 1M)")
    local.add_argument(
        "--target-gbps",
        type=_nonnegative_float,
        default=195.0,
        help="physical wire-rate pass threshold (default: 195)",
    )
    local.add_argument(
        "--reverse", action="store_true", help="measure eth2-to-eth1 traffic"
    )
    local.set_defaults(function=run_local)

    server = subparsers.add_parser("server", help="listen for one benchmark run")
    _add_common(server)
    server.add_argument(
        "--addresses",
        type=_addresses,
        help="local IP address for each interface (default: auto-detect)",
    )
    server.set_defaults(function=run_server)

    client = subparsers.add_parser("client", help="run and evaluate the benchmark")
    _add_common(client)
    client.add_argument(
        "--server-addresses",
        type=_addresses,
        required=True,
        help="server IP reached through each corresponding interface",
    )
    client.add_argument(
        "--local-addresses",
        type=_addresses,
        help="local IP address for each interface (default: auto-detect)",
    )
    client.add_argument(
        "--duration",
        type=_positive_int,
        default=30,
        help="measured seconds (default: 30)",
    )
    client.add_argument(
        "--omit", type=_nonnegative_int, default=5, help="warmup seconds (default: 5)"
    )
    client.add_argument(
        "--length", default="1M", help="iperf3 write size (default: 1M)"
    )
    client.add_argument(
        "--target-gbps",
        type=_nonnegative_float,
        default=200.0,
        help="aggregate pass threshold (default: 200)",
    )
    client.add_argument(
        "--minimum-nic-gbps",
        type=_nonnegative_float,
        default=90.0,
        help="per-NIC pass threshold (default: 90)",
    )
    client.add_argument(
        "--reverse", action="store_true", help="measure server-to-client traffic"
    )
    client.add_argument(
        "--skip-route-check",
        action="store_true",
        help="skip validation that each server address routes over its paired NIC",
    )
    client.set_defaults(function=run_client)
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
