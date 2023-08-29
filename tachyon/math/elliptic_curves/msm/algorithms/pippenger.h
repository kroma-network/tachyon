#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_H_

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/containers/adapters.h"
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

  Pippenger() : use_msm_window_naf_(PointTy::kNegationIsCheap) {}

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
    Prepare(bases_size);

    auto scalars_it = scalars_first;
    for (size_t i = 0; i < scalars_size; ++i, ++scalars_it) {
      scalars_[i] = scalars_it->ToBigInt();
    }

    if (use_msm_window_naf_) {
      AccumulateWindowNAFSums(std::move(bases_first));
    } else {
      AccumulateWindowSums(std::move(bases_first));
    }

    // We store the sum for the lowest window.
    ReturnTy lowest = std::move(*window_sums_.begin());
    auto view = absl::MakeConstSpan(window_sums_);
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
  void Prepare(size_t size) {
    if (cached_size_ == size) return;
    window_bits_ = ComputeWindowsBits(size);
    window_count_ = ComputeWindowsCount<ScalarField>(window_bits_);
    if (use_msm_window_naf_) {
      scalar_digits_.resize(size);
      for (std::vector<int64_t>& scalar_digit : scalar_digits_) {
        scalar_digit.resize(window_count_);
      }
      buckets_.resize(1 << (window_bits_ - 1));
    } else {
      // We don't need the "zero" bucket, so we only have 2^{window_bits_} - 1
      // buckets.
      buckets_.resize((1 << window_bits_) - 1);
    }
    window_sums_.resize(window_count_);
    scalars_.resize(size);
    cached_size_ = size;
  }

  void InitBuckets() {
    for (ReturnTy& bucket : buckets_) {
      bucket = ReturnTy::Zero();
    }
  }

  ReturnTy AccumulateBuckets(
      const ReturnTy& initial_value = ReturnTy::Zero()) const {
    ReturnTy running_sum = ReturnTy::Zero();
    ReturnTy window_sum = initial_value;

    // This is computed below for b buckets, using 2b curve additions.
    //
    // We could first normalize |buckets_| and then use mixed-addition
    // here, but that's slower for the kinds of groups we care about
    // (Short Weierstrass curves and Twisted Edwards curves).
    // In the case of Short Weierstrass curves,
    // mixed addition saves ~4 field multiplications per addition.
    // However normalization (with the inversion batched) takes ~6
    // field multiplications per element,
    // hence batch normalization is a slowdown.
    for (const auto& bucket : base::Reversed(buckets_)) {
      running_sum += bucket;
      window_sum += running_sum;
    }
    return window_sum;
  }

  template <typename BaseInputIterator>
  void AccumulateWindowNAFSums(BaseInputIterator bases_first) {
    for (size_t i = 0; i < cached_size_; ++i) {
      FillDigits(scalars_[i], window_bits_, &scalar_digits_[i]);
    }
    // TODO(chokobole): Optimize with openmp.
    for (size_t i = 0; i < window_sums_.size(); ++i) {
      InitBuckets();
      auto bases_it = bases_first;
      for (size_t j = 0; j < scalar_digits_.size(); ++j, ++bases_it) {
        const PointTy& base = *bases_it;
        int64_t scalar = scalar_digits_[j][i];
        if (0 < scalar) {
          buckets_[static_cast<uint64_t>(scalar - 1)] += base;
        } else if (0 > scalar) {
          buckets_[static_cast<uint64_t>(-scalar - 1)] -= base;
        }
      }
      window_sums_[i] = AccumulateBuckets();
    }
  }

  template <typename BaseInputIterator>
  void AccumulateWindowSums(BaseInputIterator bases_first) {
    // TODO(chokobole): Optimize with openmp.
    size_t window_offset = 0;
    for (size_t i = 0; i < window_sums_.size(); ++i) {
      ReturnTy window_sum = ReturnTy::Zero();
      InitBuckets();
      auto bases_it = bases_first;
      for (size_t j = 0; j < cached_size_; ++j, ++bases_it) {
        const BigInt<N>& scalar = scalars_[j];
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
          // (Recall that |buckets_| doesn't have a zero bucket.)
          if (idx != 0) {
            buckets_[idx - 1] += base;
          }
        }
      }
      window_sums_[i] = AccumulateBuckets(window_sum);
      window_offset += window_bits_;
    };
  }

  bool use_msm_window_naf_ = false;
  size_t cached_size_ = 0;
  size_t window_bits_ = 0;
  size_t window_count_ = 0;
  std::vector<std::vector<int64_t>> scalar_digits_;
  std::vector<ReturnTy> window_sums_;
  std::vector<ReturnTy> buckets_;
  std::vector<BigInt<N>> scalars_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_H_
