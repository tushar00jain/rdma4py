# Dual-NIC TCP line-rate experiment

This experiment drives multi-flow TCP over the physical path between `eth1`
and `eth2`. It launches one `iperf3` process per flow, pins each process to a
CPU local to its NIC, and validates the path with matched NIC hardware
counters. Separate processes also avoid the single-threaded parallel-stream
limit in `iperf3` releases before 3.16.

The `local` mode benchmarks two peer NICs on one host without root privileges
or network namespaces. `SO_BINDTODEVICE` forces the client data packets out of
the sender NIC even though Linux owns the receiver address locally. A run only
passes when physical TX and RX counters agree within 2%, preventing a kernel
loopback result from being mistaken for NIC bandwidth.

## Measured result

Measured on two 200 Gb/s mlx5 links at MTU 1500 with `iperf3` 3.18:

| Direction | Flows | TCP payload | Sender TX wire | Receiver RX wire | Retransmits |
|---|---:|---:|---:|---:|---:|
| `eth1` to `eth2` | 24 | 186.24 Gb/s | 200.52 Gb/s | **200.25 Gb/s** | 0 |
| `eth2` to `eth1` | 24 | 185.48 Gb/s | 200.24 Gb/s | **200.18 Gb/s** | 0 |

TCP payload is lower than line rate because it excludes Ethernet, IPv6, and
TCP framing. Wire rate comes from mlx5 `*_bytes_phy` and `*_packets_phy`
counters, including the serialized preamble and inter-packet gap. The default
195 Gb/s pass threshold represents 97.5% utilization of a 200 Gb/s link; these
runs measured 100.12% and 100.09% over their five-second counter windows.

## Local benchmark

Both NICs need carrier and an IPv4 or IPv6 address. They must be able to
exchange frames through their external peer or switch. Install `iperf3` and
`ethtool`; no privileged configuration is performed.

When an interface has multiple addresses, provide one explicitly for each NIC
in sender, receiver order:

```bash
python -m experiments.tcp_multiflow.benchmark local \
  --interfaces eth1,eth2 \
  --addresses 2001:db8:1::1,2001:db8:1::2 \
  --output-dir tcp_line_rate_results/local
```

Defaults reproduce the measured setup: 24 flows, 15 measured seconds, a
two-second `iperf3` warmup, and a five-second physical-counter sample after
seven additional seconds of settling. Use `--dry-run` to inspect all commands
and NUMA-aware CPU assignments without starting traffic. Use `--reverse` to
send from `eth2` to `eth1`.

The command exits zero only when the hardware-counter path check and line-rate
threshold both pass. It exits 2 for a valid run below the target. Raw client
and server JSON, process placement, topology metadata, and `summary.json` are
retained in the output directory.

For repeatable results:

- keep RSS, GRO, GSO, TSO, and checksum offload enabled;
- keep `irqbalance` active or distribute NIC IRQs across local CPUs;
- allow TCP autotuning buffers at least as large as the bandwidth-delay product;
- avoid unrelated work on the CPUs and NICs used by the benchmark;
- increase `--flows-per-nic` if the link remains CPU-bound.

Heavy retransmits indicate a real path problem. Check `ethtool -S`, MTU
consistency, congestion, and socket buffer limits before adding more flows.

## Two-host benchmark

The `server` and `client` modes retain support for two independent NIC paths
between two hosts:

```text
client eth1/address 1  <---->  server eth1/address 1
client eth2/address 2  <---->  server eth2/address 2
```

Start the one-shot listeners:

```bash
python -m experiments.tcp_multiflow.benchmark server \
  --interfaces eth1,eth2 \
  --addresses 2001:db8:1::2,2001:db8:2::2 \
  --output-dir tcp_line_rate_results/server
```

Then start the paired clients:

```bash
python -m experiments.tcp_multiflow.benchmark client \
  --interfaces eth1,eth2 \
  --local-addresses 2001:db8:1::1,2001:db8:2::1 \
  --server-addresses 2001:db8:1::2,2001:db8:2::2 \
  --flows-per-nic 8 --duration 30 --omit 5 \
  --target-gbps 200 --minimum-nic-gbps 90 \
  --output-dir tcp_line_rate_results/client
```

Permit the selected TCP port range in host and network firewalls. Restart the
server before a reverse run because each listener accepts one test.
