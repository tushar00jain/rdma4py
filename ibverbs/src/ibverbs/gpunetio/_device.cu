#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define DOCA_GPUNETIO_VERBS_MKEY_SWAPPED 0
#include <doca_gpunetio_dev_verbs_qp.cuh>

namespace {

constexpr auto kSharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
constexpr auto kHandler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
constexpr int32_t kTimedOut = -110;

__device__ __forceinline__ doca_gpu_dev_verbs_qp *qp_from_u64(uint64_t qp)
{
	return reinterpret_cast<doca_gpu_dev_verbs_qp *>(qp);
}

__device__ __forceinline__ doca_gpu_dev_verbs_addr address(uint64_t addr,
								    uint32_t key)
{
	return doca_gpu_dev_verbs_addr{addr, key};
}

template <bool IsRead, bool WithMcst, bool WithImm = false,
	  doca_gpu_dev_verbs_resource_sharing_mode Sharing = kSharing>
__device__ __forceinline__ uint64_t post_rdma(
	doca_gpu_dev_verbs_qp *qp, doca_gpu_dev_verbs_addr remote,
	doca_gpu_dev_verbs_addr local, uint64_t length,
	doca_gpu_dev_verbs_addr dump, uint32_t immediate = 0)
{
	const uint64_t chunks = (length + DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE - 1) /
		DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
	const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<Sharing>(
		qp, chunks + (WithMcst ? 1 : 0));
	uint64_t remaining = length;
	uint64_t last = base;

	for (uint64_t i = 0; i < chunks; ++i) {
		last = base + i;
		auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, last);
		const uint32_t bytes = static_cast<uint32_t>(
			remaining > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
				? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
				: remaining);
		if (IsRead) {
			doca_gpu_dev_verbs_wqe_prepare_read(
				qp, wqe, last, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
				remote.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
				remote.key,
				local.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
				local.key, bytes);
		} else {
			const bool last_chunk = i + 1 == chunks;
			doca_gpu_dev_verbs_wqe_prepare_write(
				qp, wqe, last,
				WithImm && last_chunk
					? DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE_IMM
					: DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
				DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
				WithImm && last_chunk ? immediate : 0,
				remote.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
				remote.key,
				local.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
				local.key, bytes);
		}
		remaining -= bytes;
	}
	if (WithMcst) {
		last = base + chunks;
		auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, last);
		doca_gpu_dev_verbs_wqe_prepare_dump(
			qp, wqe, last, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
			dump.addr, dump.key, 1);
	}
	doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(qp, base, last);
	doca_gpu_dev_verbs_submit<Sharing,
		DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kHandler>(qp, last + 1);
	return last;
}

template <doca_gpu_dev_verbs_resource_sharing_mode Sharing = kSharing>
__device__ __forceinline__ uint64_t post_send(
	doca_gpu_dev_verbs_qp *qp, doca_gpu_dev_verbs_addr local,
	uint64_t length)
{
	const uint64_t chunks = (length + DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE - 1) /
		DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
	const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<Sharing>(qp, chunks);
	uint64_t remaining = length;
	uint64_t last = base;
	for (uint64_t i = 0; i < chunks; ++i) {
		last = base + i;
		auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, last);
		const uint32_t bytes = static_cast<uint32_t>(
			remaining > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
				? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
				: remaining);
		doca_gpu_dev_verbs_wqe_prepare_send(
			qp, wqe, last, DOCA_GPUNETIO_MLX5_OPCODE_SEND,
			DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, 0,
			local.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
			local.key, bytes);
		remaining -= bytes;
	}
	doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(qp, base, last);
	doca_gpu_dev_verbs_submit<Sharing,
		DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kHandler>(qp, last + 1);
	return last;
}

template <doca_gpu_dev_verbs_resource_sharing_mode Sharing = kSharing>
__device__ __forceinline__ uint64_t post_recv(
	doca_gpu_dev_verbs_qp *qp, uint64_t local_addr, uint32_t lkey,
	uint64_t length)
{
	const uint64_t ticket = doca_gpu_dev_verbs_reserve_wq_slots<
		Sharing, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, 1);
	auto *wqe = doca_gpu_dev_verbs_get_rwqe_ptr(qp, ticket);
	doca_gpu_dev_verbs_wqe_prepare_recv(
		qp, wqe, local_addr, lkey, static_cast<uint32_t>(length));
	doca_gpu_dev_verbs_mark_wqes_ready<
		Sharing, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, ticket, ticket);
	doca_gpu_dev_verbs_submit<Sharing,
		DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kHandler,
		DOCA_GPUNETIO_VERBS_QP_RQ>(qp, ticket + 1);
	return ticket;
}

template <doca_gpu_dev_verbs_resource_sharing_mode Sharing,
	  doca_gpu_dev_verbs_qp_type Type>
__device__ __forceinline__ int32_t wait_until(
	doca_gpu_dev_verbs_cq *cq, uint64_t ticket, uint64_t deadline)
{
	while (true) {
		const int32_t status = doca_gpu_dev_verbs_poll_one_cq_at<
			Sharing, Type>(cq, ticket);
		if (status != EBUSY)
			return status;
		if (static_cast<int64_t>(clock64() - deadline) >= 0)
			return kTimedOut;
	}
}

template <typename T>
__device__ __forceinline__ int32_t reduce_volatile(
	uint64_t work_addr, uint64_t scratch_addr, uint32_t start,
	uint32_t count, uint32_t stride)
{
	volatile T *work = reinterpret_cast<volatile T *>(work_addr);
	volatile T *scratch = reinterpret_cast<volatile T *>(scratch_addr);
	for (uint32_t i = start; i < count; i += stride) {
		const T local = work[i];
		const T remote = scratch[i];
		work[i] = local + remote;
	}
	return 0;
}

template <typename T>
__device__ __forceinline__ int32_t reduce_fp8_volatile(
	uint64_t work_addr, uint64_t scratch_addr, uint32_t start,
	uint32_t count, uint32_t stride)
{
	volatile uint8_t *work = reinterpret_cast<volatile uint8_t *>(work_addr);
	volatile uint8_t *scratch = reinterpret_cast<volatile uint8_t *>(scratch_addr);
	for (uint32_t i = start; i < count; i += stride) {
		// NCCL Ring/Simple promotes each FP8 pair to FP16, adds there,
		// then uses the CUDA FP8 constructor's saturating conversion.
		T local;
		T remote;
		local.__x = work[i];
		remote.__x = scratch[i];
		const T sum(__hadd(__half(local), __half(remote)));
		work[i] = sum.__x;
	}
	return 0;
}

} // namespace

extern "C" __device__ uint64_t
rdma4py_gpunetio_clock64()
{
	return clock64();
}

extern "C" __device__ __forceinline__ __attribute__((used)) int32_t
rdma4py_gpunetio_fence_acquire()
{
	doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
	return 0;
}

extern "C" __device__ __forceinline__ __attribute__((used)) int32_t
rdma4py_gpunetio_reduce_volatile(
	uint64_t work_addr, uint64_t scratch_addr, uint32_t start,
	uint32_t count, uint32_t stride, int32_t dtype)
{
	switch (dtype) {
	case 0:
		return reduce_volatile<__half>(work_addr, scratch_addr, start, count, stride);
	case 1:
		return reduce_volatile<__nv_bfloat16>(work_addr, scratch_addr, start, count, stride);
	case 2:
		return reduce_volatile<float>(work_addr, scratch_addr, start, count, stride);
	case 3:
		return reduce_volatile<double>(work_addr, scratch_addr, start, count, stride);
	case 4:
	case 5:
		return reduce_volatile<uint8_t>(work_addr, scratch_addr, start, count, stride);
	case 6:
	case 7:
		return reduce_volatile<uint32_t>(work_addr, scratch_addr, start, count, stride);
	case 8:
	case 9:
		return reduce_volatile<uint64_t>(work_addr, scratch_addr, start, count, stride);
	case 10:
		return reduce_fp8_volatile<__nv_fp8_e4m3>(work_addr, scratch_addr, start, count, stride);
	case 11:
		return reduce_fp8_volatile<__nv_fp8_e5m2>(work_addr, scratch_addr, start, count, stride);
	default:
		return -22;
	}
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_put(uint64_t qp, uint64_t remote_addr, uint32_t rkey,
			 uint64_t local_addr, uint32_t lkey, uint64_t length)
{
	return post_rdma<false, false>(
		qp_from_u64(qp), address(remote_addr, rkey),
		address(local_addr, lkey), length, address(0, 0));
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_put_imm(uint64_t qp, uint64_t remote_addr, uint32_t rkey,
			     uint64_t local_addr, uint32_t lkey, uint64_t length,
			     uint32_t immediate)
{
	return post_rdma<false, false, true>(
		qp_from_u64(qp), address(remote_addr, rkey),
		address(local_addr, lkey), length, address(0, 0), immediate);
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_get(uint64_t qp, uint64_t remote_addr, uint32_t rkey,
			 uint64_t local_addr, uint32_t lkey, uint64_t length)
{
	return post_rdma<true, false>(
		qp_from_u64(qp), address(remote_addr, rkey),
		address(local_addr, lkey), length, address(0, 0));
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_get_mcst(uint64_t qp, uint64_t remote_addr, uint32_t rkey,
			      uint64_t local_addr, uint32_t lkey, uint64_t length,
			      uint64_t dump_addr, uint32_t dump_lkey)
{
	return post_rdma<true, true>(
		qp_from_u64(qp), address(remote_addr, rkey),
		address(local_addr, lkey), length, address(dump_addr, dump_lkey));
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_send(uint64_t qp, uint64_t local_addr, uint32_t lkey,
			  uint64_t length)
{
	return post_send(qp_from_u64(qp), address(local_addr, lkey), length);
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_recv(uint64_t qp, uint64_t local_addr, uint32_t lkey,
			  uint64_t length)
{
	return post_recv(qp_from_u64(qp), local_addr, lkey, length);
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_put_imm_exclusive(
	uint64_t qp, uint64_t remote_addr, uint32_t rkey,
	uint64_t local_addr, uint32_t lkey, uint64_t length,
	uint32_t immediate)
{
	return post_rdma<false, false, true,
		DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
		qp_from_u64(qp), address(remote_addr, rkey),
		address(local_addr, lkey), length, address(0, 0), immediate);
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_send_exclusive(
	uint64_t qp, uint64_t local_addr, uint32_t lkey, uint64_t length)
{
	return post_send<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
		qp_from_u64(qp), address(local_addr, lkey), length);
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_recv_exclusive(
	uint64_t qp, uint64_t local_addr, uint32_t lkey, uint64_t length)
{
	return post_recv<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
		qp_from_u64(qp), local_addr, lkey, length);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_send(uint64_t qp, uint64_t ticket)
{
	return doca_gpu_dev_verbs_poll_cq_at<kSharing,
		DOCA_GPUNETIO_VERBS_QP_SQ>(
		doca_gpu_dev_verbs_qp_get_cq_sq(qp_from_u64(qp)), ticket);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_test_send(uint64_t qp, uint64_t ticket)
{
	return doca_gpu_dev_verbs_poll_one_cq_at<kSharing,
		DOCA_GPUNETIO_VERBS_QP_SQ>(
		doca_gpu_dev_verbs_qp_get_cq_sq(qp_from_u64(qp)), ticket);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_recv(uint64_t qp, uint64_t ticket)
{
	return doca_gpu_dev_verbs_poll_cq_at<kSharing,
		DOCA_GPUNETIO_VERBS_QP_RQ>(
		doca_gpu_dev_verbs_qp_get_cq_rq(qp_from_u64(qp)), ticket);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_test_recv(uint64_t qp, uint64_t ticket)
{
	return doca_gpu_dev_verbs_poll_one_cq_at<kSharing,
		DOCA_GPUNETIO_VERBS_QP_RQ>(
		doca_gpu_dev_verbs_qp_get_cq_rq(qp_from_u64(qp)), ticket);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_send_until(uint64_t qp, uint64_t ticket,
				      uint64_t deadline)
{
	return wait_until<kSharing, DOCA_GPUNETIO_VERBS_QP_SQ>(
		doca_gpu_dev_verbs_qp_get_cq_sq(qp_from_u64(qp)), ticket, deadline);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_recv_until(uint64_t qp, uint64_t ticket,
				      uint64_t deadline)
{
	return wait_until<kSharing, DOCA_GPUNETIO_VERBS_QP_RQ>(
		doca_gpu_dev_verbs_qp_get_cq_rq(qp_from_u64(qp)), ticket, deadline);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_send_until_exclusive(
	uint64_t qp, uint64_t ticket, uint64_t deadline)
{
	return wait_until<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
		DOCA_GPUNETIO_VERBS_QP_SQ>(
		doca_gpu_dev_verbs_qp_get_cq_sq(qp_from_u64(qp)), ticket, deadline);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_recv_until_exclusive(
	uint64_t qp, uint64_t ticket, uint64_t deadline)
{
	return wait_until<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
		DOCA_GPUNETIO_VERBS_QP_RQ>(
		doca_gpu_dev_verbs_qp_get_cq_rq(qp_from_u64(qp)), ticket, deadline);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_recv_mcst(uint64_t qp, uint64_t ticket,
				    uint64_t dump_addr, uint32_t dump_lkey)
{
	auto *device_qp = qp_from_u64(qp);
	const uint64_t dump_ticket = doca_gpu_dev_verbs_reserve_wq_slots<kSharing>(
		device_qp, 1);
	auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(device_qp, dump_ticket);
	doca_gpu_dev_verbs_wqe_prepare_dump(
		device_qp, wqe, dump_ticket,
		DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, dump_addr, dump_lkey, 1);
	doca_gpu_dev_verbs_mark_wqes_ready<kSharing>(
		device_qp, dump_ticket, dump_ticket);
	doca_gpu_dev_verbs_submit<kSharing,
		DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kHandler>(
		device_qp, dump_ticket + 1);
	int status = doca_gpu_dev_verbs_poll_cq_at<
		kSharing, DOCA_GPUNETIO_VERBS_QP_SQ>(
		doca_gpu_dev_verbs_qp_get_cq_sq(device_qp), dump_ticket);
	if (status != 0)
		return status;
	return doca_gpu_dev_verbs_poll_cq_at<
		kSharing, DOCA_GPUNETIO_VERBS_QP_RQ>(
		doca_gpu_dev_verbs_qp_get_cq_rq(device_qp), ticket);
}
