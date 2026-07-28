# pyre-ignore-all-errors[11]: CuTe scalar objects are runtime annotations.

"""CuTe DSL FFI bindings for the GPUNetIO device bitcode ABI."""

from dataclasses import dataclass

from cutlass import Int32, Uint32, Uint64  # pyre-ignore[21]: Optional dependency.
from cutlass.base_dsl.ffi import (  # pyre-ignore[21]: Optional dependency.
    BitCode,
    extern,
)

from ._build import bitcode_path


@dataclass(frozen=True)
class CuTeOps:
    """Bound CuTe DSL external functions for one bitcode file.

    The ``*_exclusive`` operations omit GPU-wide QP arbitration and are valid
    only when exactly one CUDA thread owns that QP, as in the all-reduce's
    one-QP-per-channel mapping.
    """

    put: object
    put_imm: object
    get: object
    get_mcst: object
    send: object
    recv: object
    wait_send: object
    test_send: object
    wait_recv: object
    test_recv: object
    wait_recv_mcst: object
    clock64: object
    fence_acquire: object
    reduce_volatile: object
    wait_send_until: object
    wait_recv_until: object
    put_imm_exclusive: object
    send_exclusive: object
    recv_exclusive: object
    wait_send_until_exclusive: object
    wait_recv_until_exclusive: object


def bind(path=None, *, arch="sm_90") -> CuTeOps:
    """Bind GPUNetIO functions to a CuTe DSL bitcode source."""
    source = BitCode(str(bitcode_path(path, arch=arch)))

    @extern(name="rdma4py_gpunetio_put", source=source, overloaded=False)
    def put(
        qp: Uint64,
        remote_addr: Uint64,
        rkey: Uint32,
        local_addr: Uint64,
        lkey: Uint32,
        length: Uint64,
    ) -> Uint64: ...

    @extern(name="rdma4py_gpunetio_put_imm", source=source, overloaded=False)
    def put_imm(
        qp: Uint64,
        remote_addr: Uint64,
        rkey: Uint32,
        local_addr: Uint64,
        lkey: Uint32,
        length: Uint64,
        immediate: Uint32,
    ) -> Uint64: ...

    @extern(
        name="rdma4py_gpunetio_put_imm_exclusive",
        source=source,
        overloaded=False,
    )
    def put_imm_exclusive(
        qp: Uint64,
        remote_addr: Uint64,
        rkey: Uint32,
        local_addr: Uint64,
        lkey: Uint32,
        length: Uint64,
        immediate: Uint32,
    ) -> Uint64: ...

    @extern(name="rdma4py_gpunetio_get", source=source, overloaded=False)
    def get(
        qp: Uint64,
        remote_addr: Uint64,
        rkey: Uint32,
        local_addr: Uint64,
        lkey: Uint32,
        length: Uint64,
    ) -> Uint64: ...

    @extern(name="rdma4py_gpunetio_get_mcst", source=source, overloaded=False)
    def get_mcst(
        qp: Uint64,
        remote_addr: Uint64,
        rkey: Uint32,
        local_addr: Uint64,
        lkey: Uint32,
        length: Uint64,
        dump_addr: Uint64,
        dump_lkey: Uint32,
    ) -> Uint64: ...

    @extern(name="rdma4py_gpunetio_send", source=source, overloaded=False)
    def send(
        qp: Uint64, local_addr: Uint64, lkey: Uint32, length: Uint64
    ) -> Uint64: ...

    @extern(name="rdma4py_gpunetio_recv", source=source, overloaded=False)
    def recv(
        qp: Uint64, local_addr: Uint64, lkey: Uint32, length: Uint64
    ) -> Uint64: ...

    @extern(
        name="rdma4py_gpunetio_send_exclusive",
        source=source,
        overloaded=False,
    )
    def send_exclusive(
        qp: Uint64, local_addr: Uint64, lkey: Uint32, length: Uint64
    ) -> Uint64: ...

    @extern(
        name="rdma4py_gpunetio_recv_exclusive",
        source=source,
        overloaded=False,
    )
    def recv_exclusive(
        qp: Uint64, local_addr: Uint64, lkey: Uint32, length: Uint64
    ) -> Uint64: ...

    @extern(name="rdma4py_gpunetio_wait_send", source=source, overloaded=False)
    def wait_send(qp: Uint64, ticket: Uint64) -> Int32: ...

    @extern(name="rdma4py_gpunetio_test_send", source=source, overloaded=False)
    def test_send(qp: Uint64, ticket: Uint64) -> Int32: ...

    @extern(name="rdma4py_gpunetio_wait_recv", source=source, overloaded=False)
    def wait_recv(qp: Uint64, ticket: Uint64) -> Int32: ...

    @extern(name="rdma4py_gpunetio_test_recv", source=source, overloaded=False)
    def test_recv(qp: Uint64, ticket: Uint64) -> Int32: ...

    @extern(name="rdma4py_gpunetio_clock64", source=source, overloaded=False)
    def clock64() -> Uint64: ...

    @extern(
        name="rdma4py_gpunetio_fence_acquire",
        source=source,
        overloaded=False,
    )
    def fence_acquire() -> Int32: ...

    @extern(
        name="rdma4py_gpunetio_reduce_volatile",
        source=source,
        overloaded=False,
    )
    def reduce_volatile(
        work_addr: Uint64,
        scratch_addr: Uint64,
        start: Uint32,
        count: Uint32,
        stride: Uint32,
        dtype: Int32,
    ) -> Int32: ...

    @extern(
        name="rdma4py_gpunetio_wait_send_until",
        source=source,
        overloaded=False,
    )
    def wait_send_until(qp: Uint64, ticket: Uint64, deadline: Uint64) -> Int32: ...

    @extern(
        name="rdma4py_gpunetio_wait_recv_until",
        source=source,
        overloaded=False,
    )
    def wait_recv_until(qp: Uint64, ticket: Uint64, deadline: Uint64) -> Int32: ...

    @extern(
        name="rdma4py_gpunetio_wait_send_until_exclusive",
        source=source,
        overloaded=False,
    )
    def wait_send_until_exclusive(
        qp: Uint64, ticket: Uint64, deadline: Uint64
    ) -> Int32: ...

    @extern(
        name="rdma4py_gpunetio_wait_recv_until_exclusive",
        source=source,
        overloaded=False,
    )
    def wait_recv_until_exclusive(
        qp: Uint64, ticket: Uint64, deadline: Uint64
    ) -> Int32: ...

    @extern(
        name="rdma4py_gpunetio_wait_recv_mcst",
        source=source,
        overloaded=False,
    )
    def wait_recv_mcst(
        qp: Uint64,
        ticket: Uint64,
        dump_addr: Uint64,
        dump_lkey: Uint32,
    ) -> Int32: ...

    def ticketed(function):
        def call(*args):
            return Uint64(function(*args))

        return call

    def completion(function):
        def call(qp, ticket, *args):
            return function(qp, Uint64(ticket), *args)

        return call

    return CuTeOps(
        put=ticketed(put),
        put_imm=ticketed(put_imm),
        get=ticketed(get),
        get_mcst=ticketed(get_mcst),
        send=ticketed(send),
        recv=ticketed(recv),
        wait_send=completion(wait_send),
        test_send=completion(test_send),
        wait_recv=completion(wait_recv),
        test_recv=completion(test_recv),
        wait_recv_mcst=completion(wait_recv_mcst),
        clock64=clock64,
        fence_acquire=fence_acquire,
        reduce_volatile=reduce_volatile,
        wait_send_until=completion(wait_send_until),
        wait_recv_until=completion(wait_recv_until),
        put_imm_exclusive=ticketed(put_imm_exclusive),
        send_exclusive=ticketed(send_exclusive),
        recv_exclusive=ticketed(recv_exclusive),
        wait_send_until_exclusive=completion(wait_send_until_exclusive),
        wait_recv_until_exclusive=completion(wait_recv_until_exclusive),
    )


__all__ = ["CuTeOps", "bind"]
