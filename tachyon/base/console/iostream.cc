#include "tachyon/base/console/iostream.h"

#include <iostream>

namespace tachyon {
namespace base {
namespace internal {

ConsoleErrStream::ConsoleErrStream() : console_stream_(std::cerr) {
  console_stream_.Red();
  std::cerr << "[ERROR]: ";
}

ConsoleErrStream::~ConsoleErrStream() = default;

}  // namespace internal
}  // namespace base
}  // namespace tachyon
