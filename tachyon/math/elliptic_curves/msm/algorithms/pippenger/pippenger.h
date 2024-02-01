// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_PIPPENGER_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_PIPPENGER_H_

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_base.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_ctx.h"
#include "tachyon/math/elliptic_curves/msm/msm_util.h"
#include "tachyon/math/elliptic_curves/semigroups.h"

namespace tachyon::math {

// From:
// https://github.com/arkworks-rs/gemini/blob/main/src/kzg/msm/variable_base.rs#L20
template <size_t N>
void FillDigits(const BigInt<N>& scalar, size_t window_bits,
                std::vector<int64_t>* digits) {
  uint64_t radix = 1 << window_bits;

  uint64_t carry = 0;
  size_t bit_offset = 0;
  for (size_t i = 0; i < digits->size(); ++i) {
    // Construct a buffer of bits of the |scalar|, starting at
    // `bit_offset`.
    uint64_t bits = scalar.ExtractBits64(bit_offset, window_bits);

    // Read the actual coefficient value from the window
    uint64_t coeff = carry + bits;  // coeff = [0, 2^|window_bits|)

    // Recenter coefficients from [0,2^|window_bits|) to
    // [-2^|window_bits|/2, 2^|window_bits|/2)
    carry = (coeff + radix / 2) >> window_bits;
    (*digits)[i] = static_cast<int64_t>(coeff) -
                   static_cast<int64_t>(carry << window_bits);
    bit_offset += window_bits;
  }

  digits->back() += static_cast<int64_t>(carry << window_bits);
}

template <typename Point>
class Pippenger : public PippengerBase<Point> {
 public:
  using ScalarField = typename Point::ScalarField;
  using Bucket = typename PippengerBase<Point>::Bucket;

  constexpr static size_t N = ScalarField::N;

  Pippenger() : use_msm_window_naf_(Point::kNegationIsCheap) {
#if defined(TACHYON_HAS_OPENMP)
    parallel_windows_ = true;
#endif  // defined(TACHYON_HAS_OPENMP)
  }

  void SetParallelWindows(bool parallel_windows) {
    parallel_windows_ = parallel_windows;
  }

  void SetUseMSMWindowNAForTesting(bool use_msm_window_naf) {
    use_msm_window_naf_ = use_msm_window_naf;
  }

  template <typename BaseInputIterator, typename ScalarInputIterator,
            std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                         Point, ScalarField>>* = nullptr>
  bool Run(BaseInputIterator bases_first, BaseInputIterator bases_last,
           ScalarInputIterator scalars_first, ScalarInputIterator scalars_last,
           Bucket* ret) {
    size_t bases_size = std::distance(bases_first, bases_last);
    size_t scalars_size = std::distance(scalars_first, scalars_last);
    if (bases_size != scalars_size) {
      LOG(ERROR) << "bases_size and scalars_size don't match";
      return false;
    }
    ctx_ = PippengerCtx::CreateDefault<ScalarField>(scalars_size);

    std::vector<BigInt<N>> scalars;
    scalars.resize(scalars_size);
    auto scalars_it = scalars_first;
    for (size_t i = 0; i < scalars_size; ++i, ++scalars_it) {
      scalars[i] = scalars_it->ToBigInt();
    }

    std::vector<Bucket> window_sums =
        base::CreateVector(ctx_.window_count, Bucket::Zero());

    if (use_msm_window_naf_) {
      AccumulateWindowNAFSums(std::move(bases_first), scalars, &window_sums);
    } else {
      AccumulateWindowSums(std::move(bases_first), scalars, &window_sums);
    }

    *ret = PippengerBase<Point>::AccumulateWindowSums(
        absl::MakeConstSpan(window_sums), ctx_.window_bits);
    return true;
  }

 private:
  template <typename BaseInputIterator>
  void AccumulateSingleWindowNAFSum(
      BaseInputIterator bases_it,
      const std::vector<std::vector<int64_t>>& scalar_digits, size_t i,
      Bucket* window_sum, bool is_last_window) {
    size_t bucket_size;
    if (is_last_window) {
      bucket_size = 1 << ctx_.window_bits;
    } else {
      bucket_size = 1 << (ctx_.window_bits - 1);
    }
    std::vector<Bucket> buckets =
        base::CreateVector(bucket_size, Bucket::Zero());
    for (size_t j = 0; j < scalar_digits.size(); ++j, ++bases_it) {
      const Point& base = *bases_it;
      int64_t scalar = scalar_digits[j][i];
      if (0 < scalar) {
        buckets[static_cast<uint64_t>(scalar - 1)] += base;
      } else if (0 > scalar) {
        buckets[static_cast<uint64_t>(-scalar - 1)] -= base;
      }
    }
    *window_sum =
        PippengerBase<Point>::AccumulateBuckets(absl::MakeConstSpan(buckets));
  }

  template <typename BaseInputIterator>
  void AccumulateWindowNAFSums(BaseInputIterator bases_first,
                               absl::Span<const BigInt<N>> scalars,
                               std::vector<Bucket>* window_sums) {
    std::vector<std::vector<int64_t>> scalar_digits;
    scalar_digits.resize(scalars.size());
    for (std::vector<int64_t>& scalar_digit : scalar_digits) {
      scalar_digit.resize(ctx_.window_count);
    }
    for (size_t i = 0; i < scalars.size(); ++i) {
      FillDigits(scalars[i], ctx_.window_bits, &scalar_digits[i]);
    }
    if (parallel_windows_) {
      OPENMP_PARALLEL_FOR(size_t i = 0; i < ctx_.window_count; ++i) {
        AccumulateSingleWindowNAFSum(bases_first, scalar_digits, i,
                                     &(*window_sums)[i],
                                     i == ctx_.window_count - 1);
      }
    } else {
      for (size_t i = 0; i < ctx_.window_count; ++i) {
        AccumulateSingleWindowNAFSum(bases_first, scalar_digits, i,
                                     &(*window_sums)[i],
                                     i == ctx_.window_count - 1);
      }
    }
  }

  template <typename BaseInputIterator>
  void AccumulateSingleWindowSum(BaseInputIterator bases_first,
                                 absl::Span<const BigInt<N>> scalars,
                                 size_t window_offset, Bucket* out) {
    Bucket window_sum = Bucket::Zero();
    // We don't need the "zero" bucket, so we only have 2^{window_bits} - 1
    // buckets.
    std::vector<Bucket> buckets =
        base::CreateVector((1 << ctx_.window_bits) - 1, Bucket::Zero());
    auto bases_it = bases_first;
    for (size_t j = 0; j < scalars.size(); ++j, ++bases_it) {
      const BigInt<N>& scalar = scalars[j];
      if (scalar.IsZero()) continue;

      const Point& base = *bases_it;
      if (scalar.IsOne()) {
        // We only process unit scalars once in the first window.
        if (window_offset == 0) {
          window_sum += base;
        }
      } else {
        BigInt<N> scalar_tmp = scalar;
        // We right-shift by |window_offset|, thus getting rid of the lower
        // bits.
        scalar_tmp.DivBy2ExpInPlace(window_offset);

        // We mod the remaining bits by 2^{window_bits}, thus taking
        // |window_bits|.
        uint64_t idx = scalar_tmp[0] % (1 << ctx_.window_bits);

        // If the scalar is non-zero, we update the corresponding
        // bucket.
        // (Recall that |buckets| doesn't have a zero bucket.)
        if (idx != 0) {
          buckets[idx - 1] += base;
        }
      }
    }
    *out = PippengerBase<Point>::AccumulateBuckets(absl::MakeConstSpan(buckets),
                                                   window_sum);
  }

  template <typename BaseInputIterator>
  void AccumulateWindowSums(BaseInputIterator bases_first,
                            absl::Span<const BigInt<N>> scalars,
                            std::vector<Bucket>* window_sums) {
    if (parallel_windows_) {
      OPENMP_PARALLEL_FOR(size_t i = 0; i < ctx_.window_count; ++i) {
        AccumulateSingleWindowSum(bases_first, scalars, ctx_.window_bits * i,
                                  &(*window_sums)[i]);
      }
    } else {
      for (size_t i = 0; i < ctx_.window_count; ++i) {
        AccumulateSingleWindowSum(bases_first, scalars, ctx_.window_bits * i,
                                  &(*window_sums)[i]);
      }
    }
  }

  bool use_msm_window_naf_ = false;
  bool parallel_windows_ = false;
  PippengerCtx ctx_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_PIPPENGER_H_
