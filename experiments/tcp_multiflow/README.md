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

Create the development environment with a uv-managed Python 3.14:

```bash
uv venv --python 3.14 --managed-python
source .venv/bin/activate
```

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

### Payload and flow scaling

The scaling sweep measured these conservative physical wire rates in Gb/s,
taking the lower of sender TX and receiver RX. Each cell is a three-second
steady-state hardware-counter sample; bold entries meet the 195 Gb/s threshold.

| `iperf3` write | 1 flow | 2 flows | 4 flows | 8 flows | 16 flows | 24 flows |
|---|---:|---:|---:|---:|---:|---:|
| 1 KiB | 11.31 | 19.00 | 38.78 | 79.01 | 134.03 | **197.42** |
| 4 KiB | 22.96 | 47.68 | 86.47 | 155.91 | **200.58** | **200.83** |
| 16 KiB | 30.84 | 69.95 | 92.44 | 174.10 | **200.50** | **200.55** |
| 64 KiB | 33.11 | 66.75 | 128.22 | **199.63** | **199.12** | **200.73** |
| 256 KiB | 37.32 | 70.74 | 134.94 | **198.96** | **200.24** | **200.21** |
| 1 MiB | 29.97 | 62.14 | 111.33 | 164.99 | **200.58** | 194.57 |

The tuner selects the fewest flows that reach the threshold, rather than the
largest flow count:

| Write size | Tuned flows | TCP payload | Physical wire | Retransmits |
|---|---:|---:|---:|---:|
| 1 KiB | 24 | 182.57 Gb/s | 197.42 Gb/s | 0 |
| 4 KiB | 16 | 185.72 Gb/s | 200.58 Gb/s | 0 |
| 16 KiB | 16 | 185.25 Gb/s | 200.50 Gb/s | 0 |
| 64 KiB | 8 | 183.05 Gb/s | 199.63 Gb/s | 0 |
| 256 KiB | 8 | 182.83 Gb/s | 198.96 Gb/s | 0 |
| 1 MiB | 16 | 185.48 Gb/s | 200.58 Gb/s | 0 |

One isolated retransmit occurred outside the selected configurations. All 36
matrix points passed the physical-path counter check.

### TCP request/response latency

The latency benchmark sends one request at a time and waits for an identical
response, so the reported value is application-level TCP round-trip latency.
Both endpoints are pinned to NIC-local CPUs. These measurements used 10,000
iterations after 1,000 warmups, and every request and response was observed in
the physical NIC counters.

| Payload | eth1 to eth2 median | p95 | p99 | eth2 to eth1 median | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 1 B | 40.01 us | 80.73 us | 180.96 us | 38.61 us | 91.12 us | 399.29 us |
| 64 B | 39.55 us | 99.28 us | 788.01 us | 41.27 us | 97.81 us | 404.72 us |
| 1 KiB | 39.04 us | 96.16 us | 427.29 us | 38.21 us | 79.03 us | 181.65 us |
| 4 KiB | 111.08 us | 264.13 us | 1559.85 us | 113.25 us | 144.94 us | 322.21 us |
| 64 KiB | 222.09 us | 687.71 us | 1778.74 us | 209.66 us | 436.22 us | 1550.18 us |

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

## Scaling sweep

Sweep the default six write sizes and six flow counts with:

```bash
python -m experiments.tcp_multiflow.sweep \
  --interfaces eth1,eth2 \
  --addresses 2001:db8:1::1,2001:db8:1::2 \
  --output-dir tcp_line_rate_results/sweep
```

Use `--lengths` and `--flows` to replace either comma-separated matrix axis.
Each matrix point retains its full benchmark output. `sweep.json` contains the
aggregate report and selected configuration per write size, while `sweep.csv`
is convenient for plotting. A configuration is selected by taking the fewest
flows that reach `--target-gbps`; if none do, the highest verified wire rate is
selected.

## Latency benchmark

Measure TCP round-trip latency across the same physical NIC path with:

```bash
python -m experiments.tcp_multiflow.latency run \
  --interfaces eth1,eth2 \
  --addresses 2001:db8:1::1,2001:db8:1::2 \
  --sizes 1,64,1K,4K,64K \
  --output-dir tcp_line_rate_results/latency
```

The benchmark reports median, mean, p95, p99, minimum, and maximum RTT and
writes the statistics and counter deltas to `latency.json`. Use `--reverse` to
swap sender and receiver. It rejects the run if hardware counters do not
observe both sides of every request/response exchange.

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
