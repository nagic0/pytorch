#include <ATen/native/UsmShare.h>
#include <ATen/UsmAllocator.h>
#include <ATen/StorageUtils.h>
#include <c10/core/CPUAllocator.h>


namespace at::native {

DEFINE_DISPATCH(usm_share_stub);

// CPU implementation
static c10::Storage usm_share_cpu(const c10::Storage& src, const c10::Device& device) {
  TORCH_CHECK(
      device.type() == c10::DeviceType::CPU,
      "share_cpu only supports CPU target device, got: ",
      device);
  
  // For CPU, we can just return the same storage as USM is CPU accessible
  return src;
}

// Register CPU implementation
REGISTER_ARCH_DISPATCH(usm_share_stub, DEFAULT, &usm_share_cpu);
REGISTER_AVX512_DISPATCH(usm_share_stub, &usm_share_cpu);
REGISTER_AVX2_DISPATCH(usm_share_stub, &usm_share_cpu);
REGISTER_VSX_DISPATCH(usm_share_stub, &usm_share_cpu);
REGISTER_ZVECTOR_DISPATCH(usm_share_stub, &usm_share_cpu);

} // namespace at::native

