#include <ATen/ATen.h>
#include <ATen/native/UsmShare.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>

namespace at::native {

static c10::Storage storage_usm_share_cuda(
  const c10::Storage& src, const c10::Device& device) {
  // Get source storage information
  void* src_ptr = src.data_ptr().get();
  size_t src_bytes = src.nbytes();
  c10::Device src_device = src.device();
  
  TORCH_CHECK(
      src_device.type() == c10::DeviceType::CPU,
      "usm_share_cuda: source storage must be on CPU, got: ",
      src_device);
  TORCH_CHECK(
      device.type() == c10::DeviceType::CUDA,
      "usm_share_cuda: target device must be CUDA, got: ",
      device);
  
  // Handle empty storage
  if (src_bytes == 0) {
    return c10::Storage(c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        0,
        c10::GetAllocator(c10::DeviceType::CUDA),
        /* resizable */ false));
  }
  
  // Set device
  c10::cuda::CUDAGuard device_guard(device);
  
  // Normalize device to actual device index (handle device.index() == -1 case)
  c10::Device actual_device = device_guard.current_device();

  // Check device usm support
  auto device_properties = at::cuda::getDeviceProperties(actual_device.index());
  TORCH_CHECK(
      device_properties->integrated,
      "usm_share_cuda: target device does not support USM (not integrated GPU): ",
      actual_device);

  // Create shared memory using CUDA for usm devices
  // The memory will be accessible from both host and the target device
  cudaError_t err = cudaHostRegister(src_ptr, src_bytes, cudaHostRegisterDefault);
  TORCH_CHECK(
      err == cudaSuccess,
      "usm_share_cuda: cudaHostRegister failed for size ",
      src_bytes,
      " on device ",
      actual_device,
      " with error: ",
      cudaGetErrorString(err));

  void* shared_ptr = src_ptr;
  err = cudaHostGetDevicePointer(&shared_ptr, src_ptr, 0);
  TORCH_CHECK(
      err == cudaSuccess,
      "usm_share_cuda: cudaHostGetDevicePointer failed for size ",
      src_bytes,
      " on device ",
      actual_device,
      " with error: ",
      cudaGetErrorString(err));

  TORCH_CHECK(
      shared_ptr != nullptr,
      "usm_share_cuda: failed to create shared memory of size ",
      src_bytes,
      " on device ",
      actual_device);
  
  // Create a new storage with a custom deleter that also updates src metadata
  c10::StorageImpl* src_impl = src.unsafeGetStorageImpl();
  // Increment ref count of src to prevent it from being freed while the new
  // storage shares its data. The ref count will be decremented in the deleter.
  c10::raw::intrusive_ptr::incref(src_impl);

  struct DeleterContext {
    c10::StorageImpl* src_impl{};
    void* host_ptr{};  // Original CPU pointer for unregister
    void* device_ptr{};  // Device pointer (not used in deleter, just for reference)
    c10::Device device;
  };

  auto* deleter_context = new DeleterContext{src_impl, src_ptr, shared_ptr, actual_device};

  c10::DeleterFnPtr deleter = [](void* ctx) {
    auto* context = static_cast<DeleterContext*>(ctx);
    // Set device guard
    c10::cuda::CUDAGuard device_guard(context->device);
    // Unregister the host memory using the original host pointer
    cudaError_t err = cudaHostUnregister(context->host_ptr);
    if (err != cudaSuccess) {
      TORCH_WARN(
          "usm_share_cuda: cudaHostUnregister failed on device ",
          context->device,
          " with error: ",
          cudaGetErrorString(err));
    }
    // Decrement the ref count of src
    c10::raw::intrusive_ptr::decref(context->src_impl);
    delete context;
  };

  auto data_ptr = c10::DataPtr(shared_ptr, deleter_context, deleter, actual_device);

  auto new_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      src_bytes,
      std::move(data_ptr),
      c10::GetAllocator(c10::DeviceType::CUDA),
      /* resizable */ false);

  return c10::Storage(std::move(new_storage_impl));
}

REGISTER_CUDA_DISPATCH(usm_share_stub, &storage_usm_share_cuda);

} // namespace at::native