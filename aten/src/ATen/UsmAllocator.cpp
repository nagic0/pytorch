#include <ATen/UsmAllocator.h>
#include <iostream>
#include "c10/util/error.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <c10/util/Exception.h>

namespace at {

// UsmAllocator implementation
static void deleteUsmAllocator(void* ptr) {
  delete static_cast<UsmAllocator*>(ptr);
}

namespace {
// Read up to `size` bytes in a loop, handling EINTR and partial reads.
// Returns number of bytes read (may be less than `size` on EOF), or -1 on error.
static ssize_t read_full_loop(int fd, void* buf, size_t size) {
  size_t remaining = size;
  size_t offset = 0;
  while (remaining > 0) {
    ssize_t r = ::read(fd, static_cast<char*>(buf) + offset, remaining);
    if (r < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (r == 0) break; // EOF
    offset += static_cast<size_t>(r);
    remaining -= static_cast<size_t>(r);
  }
  return static_cast<ssize_t>(offset);
}

// Try to read `dio_size` bytes (for direct I/O) then fall back to reading
// `regular_size` bytes (non-DIO) if the dio attempt fails with an error.
static bool read_with_fd_fallback(int fd, void* dest, size_t dio_size, size_t regular_size, std::string_view filename) {
  ssize_t r = read_full_loop(fd, dest, dio_size);
  if (r > 0) return true;
  if (r < 0) {
    TORCH_WARN("USM read failed with provided fd for file ", filename,
               " (fd=", fd, "): ", c10::utils::str_error(errno), " (", errno, ")");
    lseek(fd, 0, SEEK_SET);
    r = read_full_loop(fd, dest, regular_size);
    if (r > 0) return true;
    TORCH_WARN("USM fallback read also failed for fd ", fd,
               ": ", c10::utils::str_error(errno), " (", errno, ")");
    return false;
  }
  // r == 0 (EOF) treated as failure for our use-case
  return false;
}

// Open file with `open_flags` and read into `dest` with sizes, falling back
// to non-DIO open/read if DIO fails.
static bool open_and_read_with_fallback(const std::string& filename, int open_flags, void* dest, size_t dio_size, size_t regular_size) {
  int fd = open(filename.c_str(), open_flags);
  if (fd < 0) return false;
  ssize_t r = read_full_loop(fd, dest, dio_size);
  if (r > 0) { ::close(fd); return true; }
  int saved_errno = errno;
  ::close(fd);

  TORCH_WARN("USM DIO read failed for file ", filename,
             ": ", c10::utils::str_error(saved_errno), " (", saved_errno, ")");

  // Fallback to regular read
  int regular_fd = open(filename.c_str(), O_RDONLY);
  if (regular_fd < 0) {
    TORCH_WARN("USM failed to open file for regular read: ", filename,
               ": ", c10::utils::str_error(errno), " (", errno, ")");
    return false;
  }
  r = read_full_loop(regular_fd, dest, regular_size);
  ::close(regular_fd);
  if (r > 0) return true;
  TORCH_WARN("USM regular read also failed for file ", filename,
             ": ", c10::utils::str_error(errno), " (", errno, ")");
  return false;
}
} // namespace

UsmAllocator::UsmAllocator(WithFd, std::string_view filename, int fd, size_t size)
  : filename_(filename.empty() ? "usmalloc" : filename)
  , size_(0) // to be filled later
#ifndef _WIN32
  , fd_(fd)
#endif
{
#ifdef _WIN32
  TORCH_CHECK(false, "UsmAllocator is not supported on Windows");
#else
  if (size == 0) {
    return;
  }

  // USM mode: use mmap + THP + DIO
  constexpr size_t kHugePageSize = 2L * 1024 * 1024; // 2MB
  size_t aligned_size = (size + kHugePageSize - 1) & ~(kHugePageSize - 1); // Align to 2MB
  
  // Allocate memory with mmap and enable THP
  base_ptr_ = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, 
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base_ptr_ == MAP_FAILED) {
    TORCH_CHECK(false, "USM: failed to mmap memory of size ", aligned_size);
  }
  
  // Enable THP (Transparent Huge Pages) - optional
  if (madvise(base_ptr_, aligned_size, MADV_HUGEPAGE) != 0) {
    TORCH_WARN("USM: madvise MADV_HUGEPAGE failed: ", c10::utils::str_error(errno), " (", errno, ")");
  }
  
  // Read file content using DIO if filename or fd is provided
  bool read_success = false;
  if (fd >= 0) {
    // Use the provided file descriptor
    int dio_flags = fcntl(fd, F_GETFL);
    if (dio_flags >= 0) {
      fcntl(fd, F_SETFL, dio_flags | O_DIRECT);
    }

    size_t read_size = std::min(size, aligned_size);
    size_t dio_aligned_read_size = (read_size + 511) & ~511;
    if (dio_aligned_read_size > aligned_size) dio_aligned_read_size = aligned_size;

    read_success = read_with_fd_fallback(fd, base_ptr_, dio_aligned_read_size, read_size, filename_);
  } else if (!filename_.empty() && filename_ != "usmalloc") {
    size_t read_size = std::min(size, aligned_size);
    size_t dio_aligned_read_size = (read_size + 511) & ~511;
    if (dio_aligned_read_size > aligned_size) dio_aligned_read_size = aligned_size;

    // Try open with O_DIRECT and fallback to regular open/read inside helper
    read_success = open_and_read_with_fallback(std::string(filename_), O_RDONLY | O_DIRECT, base_ptr_, dio_aligned_read_size, read_size);
    if (read_success) {
      // success already logged if it was fallback
    }
  } else {
    // No file to read, just return the allocated memory
    read_success = true;
  }
  
  if (!read_success && (fd >= 0 || (filename_ != "usmalloc" && !filename_.empty()))) {
    munmap(base_ptr_, aligned_size);
    base_ptr_ = nullptr;
    TORCH_CHECK(false, "USM: failed to read file content");
  }
  
  size_ = aligned_size;
  c10::reportMemoryUsageToProfiler(base_ptr_, static_cast<int64_t>(size_), 0, static_cast<size_t>(size_), c10::Device(c10::DeviceType::CPU));
#endif
}

UsmAllocator::UsmAllocator(std::string_view filename, size_t size)
  : UsmAllocator(WITH_FD, filename, -1, size)
{}

void UsmAllocator::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  if (base_ptr_ == nullptr) {
    return;
  }
#ifndef _WIN32
  if (munmap(base_ptr_, size_)) {
    TORCH_CHECK(false, "could not unmap the USM memory: ", c10::utils::str_error(errno), " (", errno, ")");
  }
#endif
}

UsmAllocator* UsmAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<UsmAllocator>(&deleteUsmAllocator);
}

at::DataPtr UsmAllocator::makeDataPtr(std::string_view filename, size_t size, size_t* actual_size_out) {
  auto* context = new UsmAllocator(filename, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteUsmAllocator, at::DeviceType::CPU};
}

at::DataPtr UsmAllocator::makeDataPtr(WithFd, const char *filename, int fd, size_t size, size_t* actual_size_out) {
  auto* context = new UsmAllocator(WITH_FD, filename ? std::string_view(filename) : std::string_view(""), fd, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteUsmAllocator, at::DeviceType::CPU};
}

UsmAllocator::~UsmAllocator() {
  UsmAllocator::close();
  c10::reportMemoryUsageToProfiler(base_ptr_, -static_cast<ptrdiff_t>(size_), 0, 0, c10::Device(c10::DeviceType::CPU));
}

} // namespace at