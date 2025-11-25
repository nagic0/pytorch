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
  size_t aligned_size = (size + 65535) & ~65535; // Align to 64KB
  
  // Allocate memory with mmap and enable THP
  // Use MAP_SHARED and ensure proper alignment for DIO
  base_ptr_ = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, 
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base_ptr_ == MAP_FAILED) {
    TORCH_CHECK(false, "USM: failed to mmap memory of size ", aligned_size);
  }
  
  // Check if base_ptr_ is properly aligned for DIO (512-byte alignment minimum)
  if (reinterpret_cast<uintptr_t>(base_ptr_) % 512 != 0) {
    TORCH_WARN("USM: base_ptr_ is not 512-byte aligned, DIO may fail. base_ptr=", base_ptr_);
  }
  
  // TODO: for test, turn off the thp
  // Enable THP (Transparent Huge Pages) - optional
  if (madvise(base_ptr_, aligned_size, MADV_NOHUGEPAGE) != 0) {
    // THP is optional, continue without error
  }
  
  // Read file content using DIO if filename or fd is provided
  bool read_success = false;
  if (fd >= 0) {
    // Use the provided file descriptor
    // Try to enable direct I/O on the existing fd (may not work)
    int dio_flags = fcntl(fd, F_GETFL);
    if (dio_flags >= 0) {
      fcntl(fd, F_SETFL, dio_flags | O_DIRECT);
    }
    
    size_t read_size = std::min(size, aligned_size);
    // For DIO, ensure read_size is 512-byte aligned
    size_t dio_aligned_read_size = (read_size + 511) & ~511;
    if (dio_aligned_read_size > aligned_size) {
      dio_aligned_read_size = aligned_size;
    }
    
    ssize_t bytes_read = read(fd, base_ptr_, dio_aligned_read_size);
    if (bytes_read < 0) {
      TORCH_WARN("USM read failed with provided fd for file ", filename_, 
                 " (fd=", fd, ", base_ptr=", base_ptr_, 
                 ", size=", size, ", aligned_size=", aligned_size, 
                 ", read_size=", read_size, ", dio_aligned_read_size=", dio_aligned_read_size, "): ", 
                 c10::utils::str_error(errno), " (", errno, ")");
      
      // Try without DIO by reading with original size
      lseek(fd, 0, SEEK_SET); // Reset file position
      bytes_read = read(fd, base_ptr_, read_size);
      if (bytes_read >= 0) {
        read_success = true;
      } else {
        TORCH_WARN("USM fallback read also failed for fd ", fd, 
                   ": ", c10::utils::str_error(errno), " (", errno, ")");
      }
    } else {
      read_success = true;
    }
  } else if (!filename_.empty() && filename_ != "usmalloc") {
    // Open file with DIO
    int file_fd = open(filename_.c_str(), O_RDONLY | O_DIRECT);
    if (file_fd >= 0) {
      size_t read_size = std::min(size, aligned_size);
      // For DIO, ensure read_size is 512-byte aligned
      size_t dio_aligned_read_size = (read_size + 511) & ~511;
      if (dio_aligned_read_size > aligned_size) {
        dio_aligned_read_size = aligned_size;
      }
      
      ssize_t bytes_read = read(file_fd, base_ptr_, dio_aligned_read_size);
      if (bytes_read < 0) {
        // DIO failed, log error and try fallback
        int dio_errno = errno;
        ::close(file_fd);
        TORCH_WARN("USM DIO read failed for file ", filename_, 
                   " (fd=", file_fd, ", base_ptr=", base_ptr_, 
                   ", size=", size, ", aligned_size=", aligned_size, 
                   ", read_size=", read_size, ", dio_aligned_read_size=", dio_aligned_read_size, "): ", 
                   c10::utils::str_error(dio_errno), " (", dio_errno, ")");
        
        // Fallback to regular file read
        int regular_fd = open(filename_.c_str(), O_RDONLY);
        if (regular_fd >= 0) {
          bytes_read = read(regular_fd, base_ptr_, read_size);
          ::close(regular_fd);
          if (bytes_read >= 0) {
            read_success = true;
            TORCH_WARN("USM fallback to regular read succeeded for file ", filename_, 
                       " (bytes_read=", bytes_read, ")");
          } else {
            TORCH_WARN("USM regular read also failed for file ", filename_, 
                       ": ", c10::utils::str_error(errno), " (", errno, ")");
          }
        } else {
          TORCH_WARN("USM failed to open file for regular read: ", filename_, 
                     ": ", c10::utils::str_error(errno), " (", errno, ")");
        }
      } else {
        ::close(file_fd);
        read_success = true;
      }
    } else {
      // DIO open failed, try regular file read
      int dio_errno = errno;
      TORCH_WARN("USM failed to open file with DIO: ", filename_, 
                 ": ", c10::utils::str_error(dio_errno), " (", dio_errno, ")");
      
      // Fallback to regular file read if DIO fails
      int regular_fd = open(filename_.c_str(), O_RDONLY);
      if (regular_fd >= 0) {
        size_t read_size = std::min(size, aligned_size);
        ssize_t bytes_read = read(regular_fd, base_ptr_, read_size);
        ::close(regular_fd);
        if (bytes_read >= 0) {
          read_success = true;
          TORCH_WARN("USM fallback to regular read succeeded for file ", filename_, 
                     " (bytes_read=", bytes_read, ")");
        } else {
          TORCH_WARN("USM regular read failed for file ", filename_, 
                     ": ", c10::utils::str_error(errno), " (", errno, ")");
        }
      } else {
        TORCH_WARN("USM failed to open file for regular read: ", filename_, 
                   ": ", c10::utils::str_error(errno), " (", errno, ")");
      }
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