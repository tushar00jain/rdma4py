# pyre-ignore-all-errors[21]: Test dependency and experiment package.
"""Tests for the striped tensor TCP transport."""

import multiprocessing
import socket
import threading
from types import SimpleNamespace

import pytest
from experiments.tcp_multiflow import transport


def _free_port_range(flows):
    for base_port in range(30000, 60000):
        sockets = []
        try:
            for flow in range(flows):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("127.0.0.1", base_port + flow))
                sockets.append(sock)
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
        if len(sockets) == flows:
            return base_port
    raise RuntimeError("could not reserve a test port range")


def test_partitions_cover_every_byte_once():
    partitions = transport._partitions(101, 8)

    assert partitions[0][0] == 0
    assert partitions[-1][1] == 101
    assert all(left[1] == right[0] for left, right in zip(partitions, partitions[1:]))
    assert max(end - start for start, end in partitions) == 13


def test_tensor_buffer_accepts_buffer_and_duck_typed_cpu_tensor():
    direct = bytearray(b"tensor")
    assert transport._tensor_buffer(direct, writable=True).view.tobytes() == b"tensor"

    tensor = SimpleNamespace(
        detach=lambda: tensor,
        device=SimpleNamespace(type="cpu"),
        is_contiguous=lambda: True,
        numpy=lambda: direct,
    )
    assert transport._tensor_buffer(tensor, writable=True).owner is direct


def test_numpy_tensor_over_shared_storage_selects_process_workers():
    pytest.importorskip("numpy")
    storage = transport.SharedTensor(1024)
    tensor = storage.numpy((256,), dtype="uint32")

    assert transport._tensor_buffer(tensor, writable=True).shared

    del tensor
    storage.close()


def test_tensor_buffer_rejects_cuda_and_readonly_destination():
    cuda = SimpleNamespace(
        detach=lambda: cuda,
        device=SimpleNamespace(type="cuda"),
        is_contiguous=lambda: True,
        numpy=lambda: bytearray(1),
    )

    with pytest.raises(ValueError, match="CPU tensor"):
        transport._tensor_buffer(cuda, writable=False)
    with pytest.raises(ValueError, match="read-only"):
        transport._tensor_buffer(b"read only", writable=True)


def test_sender_and_receiver_transfer_into_preallocated_buffer(monkeypatch):
    monkeypatch.setattr(transport, "_bind_device", lambda sock, interface: None)
    flows = 3
    base_port = _free_port_range(flows)
    endpoint = transport.NetworkEndpoint("lo", "127.0.0.1")
    receiver = transport.TensorReceiver(
        endpoint, flows=flows, base_port=base_port, chunk_bytes=257
    )
    sender = transport.TensorSender(
        endpoint,
        "127.0.0.1",
        flows=flows,
        base_port=base_port,
        chunk_bytes=257,
    )
    source = bytearray((index * 17) & 0xFF for index in range(100_003))
    destination = bytearray(len(source))
    ready = threading.Event()
    received = []
    errors = []

    def receive():
        try:
            received.append(receiver.receive_into(destination, ready_event=ready))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=receive)
    thread.start()
    assert ready.wait(timeout=5)
    sent = sender.send(source)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    assert destination == source
    assert sent.nbytes == len(source)
    assert len(sent.flows) == flows
    assert received[0].nbytes == len(source)


def test_process_workers_transfer_into_shared_tensor(monkeypatch):
    monkeypatch.setattr(transport, "_bind_device", lambda sock, interface: None)
    flows = 3
    base_port = _free_port_range(flows)
    endpoint = transport.NetworkEndpoint("lo", "127.0.0.1")
    destination = transport.SharedTensor(1_000_003)
    source = transport.SharedTensor(destination.nbytes)
    expected = bytearray((index * 29) & 0xFF for index in range(destination.nbytes))
    source.buffer[:] = expected
    receiver = transport.TensorReceiver(
        endpoint,
        flows=flows,
        base_port=base_port,
        chunk_bytes=4096,
        worker_mode="process",
    )
    sender = transport.TensorSender(
        endpoint,
        "127.0.0.1",
        flows=flows,
        base_port=base_port,
        chunk_bytes=4096,
        worker_mode="process",
    )
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    reader, writer = context.Pipe(duplex=False)

    def receive():
        try:
            writer.send(
                (
                    True,
                    receiver.receive_into(destination, iterations=3, ready_event=ready),
                )
            )
        except BaseException as error:
            ready.set()
            writer.send((False, str(error)))
        finally:
            writer.close()

    process = context.Process(target=receive)
    process.start()
    writer.close()
    assert ready.wait(timeout=5)
    sent = sender.send(source, iterations=3)
    succeeded, result = reader.recv()
    process.join(timeout=5)

    assert not process.is_alive()
    assert succeeded, result
    assert result.nbytes == destination.nbytes * 3
    assert destination.buffer.tobytes() == expected
    assert sent.nbytes == destination.nbytes * 3
    assert sum(flow.nbytes for flow in sent.flows) == sent.nbytes
    assert result.completion_seconds >= result.seconds
    reader.close()
    source.close()
    destination.close()
