#ifndef TACHYON_MATH_BASE_INVALID_OPERATION_H_
#define TACHYON_MATH_BASE_INVALID_OPERATION_H_

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/logging.h"

namespace tachyon::math {

bool inline InvalidOperation(bool invalid, std::string_view msg) {
  if (UNLIKELY(invalid)) {
    // TODO(ashjeong): implement CUDA error logging
#if !TACHYON_CUDA
    LOG(ERROR) << msg;
#endif  // TACHYON_CUDA
    return true;
  }
  return false;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_INVALID_OPERATION_H_
