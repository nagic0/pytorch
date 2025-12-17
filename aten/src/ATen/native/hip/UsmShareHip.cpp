#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceType.h>
#include <hip/hip_runtime.h>

namespace at::native {

// Implementation for HIP backend
// self: dummy tensor on HIP device (carries device info)
// src:  CPU storage containing the data
Tensor usm_share_from_hip(const Tensor& self, const c10::Storage& src) {
  void* src_ptr = src.data_ptr();
  size_t src_bytes = src.nbytes();
  c10::Device target_device = self.device();

  TORCH_CHECK(
      src.device().is_cpu(),
      "usm_share_from_hip: source storage must be on CPU, got: ",
      src.device());
  TORCH_CHECK(
      target_device.type() == c10::DeviceType::HIP,
      "usm_share_from_hip: target device must be HIP, got: ",
      target_device);

  // Handle empty case
  if (src_bytes == 0) {
    return at::empty_like(self, self.options());
  }

  // Set device guard
  c10::hip::HIPGuard device_guard(target_device.index());

  // Register CPU memory with HIP
  hipError_t err = hipHostRegister(src_ptr, src_bytes, hipHostRegisterDefault);
  TORCH_CHECK(
      err == hipSuccess,
      "hipHostRegister failed: ",
      hipGetErrorString(err));

  // Get device pointer for the registered host memory
  void* dev_ptr = nullptr;
  err = hipHostGetDevicePointer(&dev_ptr, src_ptr, 0);
  TORCH_CHECK(
      err == hipSuccess,
      "hipHostGetDevicePointer failed: ",
      hipGetErrorString(err));

  // Create deleter to unregister memory when storage is destroyed
  auto deleter = [src_ptr](void* /* data */) {
    hipError_t err = hipHostUnregister(src_ptr);
    if (err != hipSuccess) {
      // Log error but don't throw in destructor
      fprintf(stderr, "hipHostUnregister failed: %s\n", hipGetErrorString(err));
    }
  };

  // Create StorageImpl with custom deleter
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      src_bytes,
      c10::DataPtr(dev_ptr, src_ptr, deleter, target_device),
      at::getCaching HIPAllocator(),
      false /* resizable */);

  // Wrap in a Tensor
  return at::empty({0}, self.options()).set_(std::move(storage_impl));
}

} // namespace at::native
