#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>

namespace at::native {

// Implementation for CUDA backend
// self: dummy tensor on CUDA device (carries device info)
// src:  CPU storage containing the data
Tensor usm_share_from_cuda(const Tensor& self, const c10::Storage& src) {
  void* src_ptr = src.data_ptr();
  size_t src_bytes = src.nbytes();
  c10::Device target_device = self.device();

  TORCH_CHECK(
      src.device().is_cpu(),
      "usm_share_from_cuda: source storage must be on CPU, got: ",
      src.device());
  TORCH_CHECK(
      target_device.type() == c10::DeviceType::CUDA,
      "usm_share_from_cuda: target device must be CUDA, got: ",
      target_device);

  // Handle empty case
  if (src_bytes == 0) {
    return at::empty_like(self, self.options());
  }

  // Set device guard
  c10::cuda::CUDAGuard device_guard(target_device);
  c10::Device actual_device = device_guard.current_device();

  // Check device USM support (integrated GPU only)
  auto device_properties = at::cuda::getDeviceProperties(actual_device.index());
  TORCH_CHECK(
      device_properties->integrated,
      "usm_share_from_cuda: target device does not support USM (requires integrated GPU): ",
      actual_device);

  // Register host memory for CUDA access
  cudaError_t err = cudaHostRegister(src_ptr, src_bytes, cudaHostRegisterDefault);
  TORCH_CHECK(
      err == cudaSuccess,
      "usm_share_from_cuda: cudaHostRegister failed, error: ",
      cudaGetErrorString(err));

  // Get device pointer
  void* dev_ptr = src_ptr;
  err = cudaHostGetDevicePointer(&dev_ptr, src_ptr, 0);
  TORCH_CHECK(
      err == cudaSuccess,
      "usm_share_from_cuda: cudaHostGetDevicePointer failed, error: ",
      cudaGetErrorString(err));

  // Create custom deleter
  auto deleter = [src_ptr](void* p) {
    cudaError_t err = cudaHostUnregister(src_ptr);
    if (err != cudaSuccess) {
      TORCH_WARN(
          "usm_share_from_cuda: cudaHostUnregister failed, error: ",
          cudaGetErrorString(err));
    }
  };

  // Create storage with device pointer
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      src_bytes,
      c10::DataPtr(dev_ptr, src_ptr, deleter, actual_device),
      c10::GetAllocator(c10::DeviceType::CUDA),
      false);

  return at::empty({0}, self.options()).set_(std::move(storage_impl));
}

} // namespace at::native
