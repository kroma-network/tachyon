#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_CTX_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_CTX_H_

#include <stddef.h>

#include <cmath>

#include "tachyon/export.h"

namespace tachyon::math {

struct TACHYON_EXPORT MSMCtx {
  unsigned int window_count = 0;
  unsigned int window_bits = 0;
  unsigned int size = 0;

  constexpr unsigned int GetWindowLength() const { return 1 << window_bits; }

  template <typename ScalarField>
  constexpr static MSMCtx CreateDefault(size_t size) {
    MSMCtx ctx;
    ctx.window_bits = ComputeWindowsBits(size);
    ctx.window_count = ComputeWindowsCount<ScalarField>(ctx.window_bits);
    ctx.size = size;
    return ctx;
  }

  // The result of this function is only approximately `ln(a)`.
  // See https://github.com/scipr-lab/zexe/issues/79#issue-556220473
  constexpr static unsigned int LnWithoutFloats(size_t a) {
    // log2(a) * ln(2)
    return log2(a) * 69 / 100;
  }

  constexpr static unsigned int ComputeWindowsBits(size_t size) {
    if (size < 32) {
      return 3;
    } else {
      return LnWithoutFloats(size) + 2;
    }
  }

  template <typename ScalarField>
  constexpr static unsigned int ComputeWindowsCount(unsigned int window_bits) {
    return (ScalarField::Config::kModulusBits + window_bits - 1) / window_bits;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_CTX_H_
