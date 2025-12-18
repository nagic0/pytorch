#include <ATen/ATen.h>
#include <ATen/mps/MPSDevice.h>
#include <Metal/Metal.h>

namespace at::native {

// Implementation for MPS backend
// src passed by value to match dispatcher signature
Tensor usm_share_from_mps(const Tensor& self, c10::Storage src) {
  void* ptr = src.data_ptr().get();
  size_t size = src.nbytes();

  TORCH_CHECK(
      src.device().is_cpu(),
      "usm_share_from_mps: source storage must be on CPU, got: ",
      src.device());
  TORCH_CHECK(
      self.device().type() == c10::DeviceType::MPS,
      "usm_share_from_mps: target device must be MPS, got: ",
      self.device());

  // Handle empty case
  if (size == 0) {
    return at::empty_like(self, self.options());
  }

  // 1. Check Alignment (Critical for Metal NoCopy)
  // vm_page_size is typically 16KB on Apple Silicon
  TORCH_CHECK((uintptr_t)ptr % 16384 == 0,
      "usm_share_from_mps: MPS requires CPU pointer to be 16KB aligned. "
      "Use torch.empty(..., pin_memory=True) or posix_memalign.");

  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();

  // 2. Create NoCopy Buffer
  id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:ptr
                                                   length:size
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
  
  // Retain buffer to keep it alive as long as Storage exists
  CFRetain((CFTypeRef)buffer);
  
  auto deleter = [buffer](void* p) {
    CFRelease((CFTypeRef)buffer);
  };

  // 3. Create StorageImpl
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      c10::DataPtr([buffer contents], buffer, deleter, self.device()),
      c10::GetAllocator(c10::DeviceType::MPS),
      false);

  return at::empty({0}, self.options()).set_(std::move(storage_impl));
}

} // namespace at::native
