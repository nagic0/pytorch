#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

namespace at::native {

// Implementation for HIP backend
// self: dummy tensor on HIP (unused except for context)
// src:  wrapped tensor on CPU
Tensor usm_share_from_hip(const Tensor& self, const Tensor& src) {
  void* ptr = src.data_ptr();
  size_t size = src.nbytes();
  int device_id = self.device().index();

  TORCH_CHECK(
      src.device().is_cpu(),
      "usm_share_from_hip: source tensor must be on CPU, got: ",
      src.device());
  TORCH_CHECK(
      self.device().type() == c10::DeviceType::HIP,
      "usm_share_from_hip: target device must be HIP, got: ",
      self.device());

  // Handle empty case
  if (size == 0) {
    return at::empty_like(self, self.options());
  }

  // Ensure we are on the correct device
  hipSetDevice(device_id);

  // 1. Register Host Memory
  hipError_t err = hipHostRegister(ptr, size, hipHostRegisterMapped);
  TORCH_CHECK(err == hipSuccess, "hipHostRegister failed with error: ", err);

  // 2. Get Device Pointer
  void* dev_ptr = nullptr;
  err = hipHostGetDevicePointer(&dev_ptr, ptr, 0);
  if (err != hipSuccess) {
    hipHostUnregister(ptr);
    TORCH_CHECK(false, "hipHostGetDevicePointer failed");
  }

  // 3. Create Deleter (Unregister when done)
  auto deleter = [ptr](void* p) {
    hipHostUnregister(ptr);
  };

  // 4. Create new StorageImpl wrapping the device pointer
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      c10::DataPtr(dev_ptr, ptr, deleter, self.device()),
      c10::GetAllocator(c10::DeviceType::HIP),
      false);

  // 5. Return a new Tensor wrapping this storage
  return at::empty({0}, self.options()).set_(std::move(storage_impl));
}

} // namespace at::native
