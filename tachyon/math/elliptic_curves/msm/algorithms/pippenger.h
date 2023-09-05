#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_H_

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger_util.h"
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

template <typename PointTy>
class Pippenger {
 public:
  using ScalarField = typename PointTy::ScalarField;
  using ReturnTy =
      typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;

  constexpr static size_t N = ScalarField::N;

  Pippenger() : use_msm_window_naf_(PointTy::kNegationIsCheap) {
#if defined(TACHYON_HAS_OPENMP)
    parallel_windows_ = true;
#endif  // defined(TACHYON_HAS_OPENMP)
  }

  void SetParallelWindows(bool parallel_windows) {
    parallel_windows_ = parallel_windows;
#if !defined(TACHYON_HAS_OPENMP)
    LOG_IF(WARNING, parallel_windows) << "Set parallel windows without openmp";
#endif  // !defined(TACHYON_HAS_OPENMP)
  }

  void SetUseMSMWindowNAForTesting(bool use_msm_window_naf) {
    use_msm_window_naf_ = use_msm_window_naf;
  }

  template <typename BaseInputIterator, typename ScalarInputIterator,
            std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                         PointTy, ScalarField>>* = nullptr>
  bool Run(BaseInputIterator bases_first, BaseInputIterator bases_last,
           ScalarInputIterator scalars_first, ScalarInputIterator scalars_last,
           ReturnTy* ret) {
    size_t bases_size = std::distance(bases_first, bases_last);
    size_t scalars_size = std::distance(scalars_first, scalars_last);
    if (bases_size != scalars_size) {
      LOG(ERROR) << "bases_size and scalars_size don't match";
      return false;
    }
    window_bits_ = ComputeWindowsBits(scalars_size);
    window_count_ = ComputeWindowsCount<ScalarField>(window_bits_);

    std::vector<BigInt<N>> scalars;
    scalars.resize(scalars_size);
    auto scalars_it = scalars_first;
    for (size_t i = 0; i < scalars_size; ++i, ++scalars_it) {
      scalars[i] = scalars_it->ToBigInt();
    }

    std::vector<ReturnTy> window_sums =
        base::CreateVector(window_count_, ReturnTy::Zero());

    if (use_msm_window_naf_) {
      AccumulateWindowNAFSums(std::move(bases_first), scalars, &window_sums);
    } else {
      AccumulateWindowSums(std::move(bases_first), scalars, &window_sums);
    }

    // We store the sum for the lowest window.
    ReturnTy lowest = std::move(window_sums.front());
    auto view = absl::MakeConstSpan(window_sums);
    view.remove_prefix(1);

    // We're traversing windows from high to low.
    *ret =
        lowest + std::accumulate(view.rbegin(), view.rend(), ReturnTy::Zero(),
                                 [this](ReturnTy& total, const ReturnTy& sum) {
                                   total += sum;
                                   for (size_t i = 0; i < window_bits_; ++i) {
                                     total.DoubleInPlace();
                                   }
                                   return total;
                                 });
    return true;
  }

 private:
  ReturnTy AccumulateBuckets(
      const std::vector<ReturnTy>& buckets,
      const ReturnTy& initial_value = ReturnTy::Zero()) const {
    ReturnTy running_sum = ReturnTy::Zero();
    ReturnTy window_sum = initial_value;

    // This is computed below for b buckets, using 2b curve additions.
    //
    // We could first normalize |buckets| and then use mixed-addition
    // here, but that's slower for the kinds of groups we care about
    // (Short Weierstrass curves and Twisted Edwards curves).
    // In the case of Short Weierstrass curves,
    // mixed addition saves ~4 field multiplications per addition.
    // However normalization (with the inversion batched) takes ~6
    // field multiplications per element,
    // hence batch normalization is a slowdown.
    for (const auto& bucket : base::Reversed(buckets)) {
      running_sum += bucket;
      window_sum += running_sum;
    }
    return window_sum;
  }

  template <typename BaseInputIterator>
  void AccumulateSingleWindowNAFSum(
      BaseInputIterator bases_it,
      const std::vector<std::vector<int64_t>>& scalar_digits, size_t i,
      ReturnTy* window_sum, bool is_last_window) {
    size_t bucket_size;
    if (is_last_window) {
      bucket_size = 1 << window_bits_;
    } else {
      bucket_size = 1 << (window_bits_ - 1);
    }
    std::vector<ReturnTy> buckets =
        base::CreateVector(bucket_size, ReturnTy::Zero());
    for (size_t j = 0; j < scalar_digits.size(); ++j, ++bases_it) {
      const PointTy& base = *bases_it;
      int64_t scalar = scalar_digits[j][i];
      if (0 < scalar) {
        buckets[static_cast<uint64_t>(scalar - 1)] += base;
      } else if (0 > scalar) {
        buckets[static_cast<uint64_t>(-scalar - 1)] -= base;
      }
    }
    *window_sum = AccumulateBuckets(buckets);
  }

  template <typename BaseInputIterator>
  void AccumulateWindowNAFSums(BaseInputIterator bases_first,
                               absl::Span<const BigInt<N>> scalars,
                               std::vector<ReturnTy>* window_sums) {
    std::vector<std::vector<int64_t>> scalar_digits;
    scalar_digits.resize(scalars.size());
    for (std::vector<int64_t>& scalar_digit : scalar_digits) {
      scalar_digit.resize(window_count_);
    }
    for (size_t i = 0; i < scalars.size(); ++i) {
      FillDigits(scalars[i], window_bits_, &scalar_digits[i]);
    }
    if (parallel_windows_) {
      OPENMP_PARALLEL_FOR(size_t i = 0; i < window_count_; ++i) {
        AccumulateSingleWindowNAFSum(bases_first, scalar_digits, i,
                                     &(*window_sums)[i],
                                     i == window_count_ - 1);
      }
    } else {
      for (size_t i = 0; i < window_count_; ++i) {
        AccumulateSingleWindowNAFSum(bases_first, scalar_digits, i,
                                     &(*window_sums)[i],
                                     i == window_count_ - 1);
      }
    }
  }

  template <typename BaseInputIterator>
  void AccumulateSingleWindowSum(BaseInputIterator bases_first,
                                 absl::Span<const BigInt<N>> scalars,
                                 size_t window_offset, ReturnTy* out) {
    ReturnTy window_sum = ReturnTy::Zero();
    // We don't need the "zero" bucket, so we only have 2^{window_bits_} - 1
    // buckets.
    std::vector<ReturnTy> buckets =
        base::CreateVector((1 << window_bits_) - 1, ReturnTy::Zero());
    auto bases_it = bases_first;
    for (size_t j = 0; j < scalars.size(); ++j, ++bases_it) {
      const BigInt<N>& scalar = scalars[j];
      if (scalar.IsZero()) continue;

      const PointTy& base = *bases_it;
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

        // We mod the remaining bits by 2^{window_bits_}, thus taking
        // |window_bits_|.
        uint64_t idx = scalar_tmp[0] % (1 << window_bits_);

        // If the scalar is non-zero, we update the corresponding
        // bucket.
        // (Recall that |buckets| doesn't have a zero bucket.)
        if (idx != 0) {
          buckets[idx - 1] += base;
        }
      }
    }
    *out = AccumulateBuckets(buckets, window_sum);
  }

  template <typename BaseInputIterator>
  void AccumulateWindowSums(BaseInputIterator bases_first,
                            absl::Span<const BigInt<N>> scalars,
                            std::vector<ReturnTy>* window_sums) {
    if (parallel_windows_) {
      OPENMP_PARALLEL_FOR(size_t i = 0; i < window_count_; ++i) {
        AccumulateSingleWindowSum(bases_first, scalars, window_bits_ * i,
                                  &(*window_sums)[i]);
      }
    } else {
      for (size_t i = 0; i < window_count_; ++i) {
        AccumulateSingleWindowSum(bases_first, scalars, window_bits_ * i,
                                  &(*window_sums)[i]);
      }
    }
  }

  bool use_msm_window_naf_ = false;
  bool parallel_windows_ = false;
  size_t window_bits_ = 0;
  size_t window_count_ = 0;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_H_
