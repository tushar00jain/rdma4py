# Dual-NIC TCP line-rate experiment

This experiment drives multi-flow TCP over the physical path between `eth1`
and `eth2`. The Python engine launches one sender and one receiver process per
flow, uses Python's `socket.sendall` and `socket.recv_into` as the data path,
and pins each process to a CPU local to its NIC. An `iperf3` engine remains
available as a reference. Both engines validate the path with matched NIC
hardware counters.

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

### iperf3 reference

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

### Python payload and flow scaling

The scaling sweep measured these conservative physical wire rates in Gb/s,
taking the lower of sender TX and receiver RX. Each cell is a three-second
steady-state hardware-counter sample from the multi-process Python engine;
bold entries meet the 195 Gb/s threshold.

| Python write | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 flows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 KiB | 9.94 | 19.39 | 34.29 | 67.00 | 119.46 | 193.47 | **197.98** | **200.71** |
| 4 KiB | 20.86 | 42.95 | 87.48 | 130.79 | 193.06 | **199.07** | **198.75** | **200.95** |
| 16 KiB | 24.44 | 42.87 | 74.98 | 145.68 | 191.69 | **197.20** | **200.53** | **200.97** |
| 64 KiB | 31.16 | 39.36 | 82.17 | 150.02 | 194.48 | 194.89 | **198.48** | 175.59 |
| 256 KiB | 17.77 | 30.42 | 70.52 | 127.70 | **196.87** | 182.46 | **200.47** | 181.88 |
| 1 MiB | 18.99 | 40.77 | 70.78 | 123.07 | 191.95 | **199.18** | **198.59** | 194.62 |

All 48 points passed the physical-path counter check. Scaling is not monotonic:
once the link is saturated, extra Python processes can add enough scheduling
and cache pressure to reduce throughput.

The final choices were rerun for 15 measured seconds with a five-second
steady-state counter sample. The short sweep's first passing result was
promoted to a larger flow count when it did not sustain the threshold:

| Write size | Confirmed flows | TCP payload | Physical wire | Retransmits |
|---|---:|---:|---:|---:|
| 1 KiB | 32 | 184.85 Gb/s | 199.41 Gb/s | 0 |
| 4 KiB | 32 | 171.80 Gb/s | 196.16 Gb/s | 5 |
| 16 KiB | 48 | 184.86 Gb/s | 198.95 Gb/s | 1 |
| 64 KiB | 32 | 184.34 Gb/s | 199.20 Gb/s | 0 |
| 256 KiB | 16 | 180.70 Gb/s | 195.65 Gb/s | 0 |
| 1 MiB | 24 | 184.30 Gb/s | 198.73 Gb/s | 2 |

A reverse Python spot check with 1 MiB writes and 24 flows sustained
196.28 Gb/s from `eth2` to `eth1` with one retransmit.

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

## iperf3 reference benchmark

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

## Python throughput benchmark

Run one Python sender and receiver process per TCP flow with:

```bash
python -m experiments.tcp_multiflow.python_benchmark local \
  --interfaces eth1,eth2 \
  --addresses 2001:db8:1::1,2001:db8:1::2 \
  --flows-per-nic 32 --length 64K \
  --output-dir tcp_line_rate_results/python-local
```

This path does not invoke `iperf3`. Each sender repeatedly calls
`socket.sendall` with the requested write size, while its paired receiver uses
`socket.recv_into`. Separate processes bypass the GIL and are assigned to
different NIC-local CPUs. Linux `TCP_INFO` supplies per-flow retransmit counts.
The exit status and physical-counter validation match the reference benchmark.

## Scaling sweep

Sweep the default six write sizes and eight flow counts with the Python engine:

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
selected. `--resume` reuses completed points after an interrupted or expanded
run. Pass `--engine iperf3` only when an iperf3 comparison is desired.

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
