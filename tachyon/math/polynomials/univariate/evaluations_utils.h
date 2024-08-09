// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_EVALUATIONS_UTILS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_EVALUATIONS_UTILS_H_

#include <utility>
#include <vector>

#include "tachyon/base/bits.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/profiler.h"

namespace tachyon::math {

template <typename F>
std::vector<F> SwapBitRevElements(const std::vector<F>& vals) {
  size_t n = vals.size();
  std::vector<F> ret = vals;
  SwapBitRevElementsInPlace(ret, n, base::bits::CheckedLog2(n));
  return ret;
}

// Swaps the elements at each index with the element at its bit-reversed index
// in place.
// ex. For a vector of size 8, the element at index 1(001) and the
//     element at index 4(100) are swapped.
template <typename Container>
void SwapBitRevElementsInPlace(Container& container, size_t size,
                               size_t log_len) {
  TRACE_EVENT("Utils", "SwapBitRevElementsInPlace");
  if (size <= 1) return;
  OMP_PARALLEL_FOR(size_t idx = 1; idx < size; ++idx) {
    size_t ridx = base::bits::ReverseBitsLen(idx, log_len);
    if (idx < ridx) {
      std::swap(container.at(idx), container.at(ridx));
    }
  }
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_EVALUATIONS_UTILS_H_
