#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/elliptic_curves/msm/msm_util.h"
#include "tachyon/math/elliptic_curves/semigroups.h"

namespace tachyon::math {

// From:
// https://github.com/arkworks-rs/gemini/blob/main/src/kzg/msm/variable_base.rs#L20
template <size_t N>
std::vector<int64_t> MakeDigits(const BigInt<N>& scalar, size_t w,
                                size_t num_bits) {
  uint64_t radix = 1 << w;
  uint64_t window_mask = radix - 1;

  uint64_t carry = 0;
  if (num_bits == 0) {
    num_bits = N * 64;
  }
  size_t digits_count = (num_bits + w - 1) / w;
  std::vector<int64_t> digits =
      base::CreateVector(digits_count, static_cast<int64_t>(0));
  for (size_t i = 0; i < digits.size(); ++i) {
    // Construct a buffer of bits of the scalar, starting at `bit_offset`.
    size_t bit_offset = i * w;
    size_t u64_idx = bit_offset / 64;
    size_t bit_idx = bit_offset % 64;

    // Read the bits from the scalar
    uint64_t bit_buf;
    if (bit_idx < 64 - w || u64_idx == N - 1) {
      // This window's bits are contained in a single u64,
      // or it's the last u64 anyway.
      bit_buf = scalar[u64_idx] >> bit_idx;
    } else {
      // Combine the current u64's bits with the bits from the next u64
      bit_buf = (scalar[u64_idx] >> bit_idx) |
                (scalar[1 + u64_idx] << (64 - bit_idx));
    }

    // Read the actual coefficient value from the window
    uint64_t coeff = carry + (bit_buf & window_mask);  // coeff = [0, 2^r)

    // Recenter coefficients from [0,2^w) to [-2^w/2, 2^w/2)
    carry = (coeff + radix / 2) >> w;
    digits[i] = static_cast<int64_t>(coeff) - static_cast<int64_t>(carry << w);
  }

  digits[digits_count - 1] += static_cast<int64_t>(carry << w);

  return digits;
}

template <typename PointTy>
class VariableBaseMSM {
 public:
  using ScalarField = typename PointTy::ScalarField;
  using ReturnTy =
      typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;

  template <typename BaseInputIterator, typename ScalarInputIterator,
            std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                         PointTy, ScalarField>>* = nullptr>
  static ReturnTy MSM(BaseInputIterator bases_first,
                      BaseInputIterator bases_last,
                      ScalarInputIterator scalars_first,
                      ScalarInputIterator scalars_last) {
    return DoMSM(std::move(bases_first), std::move(bases_last),
                 std::move(scalars_first), std::move(scalars_last),
                 PointTy::kNegationIsCheap);
  }

  template <typename BaseContainer, typename ScalarContainer>
  static ReturnTy MSM(BaseContainer&& bases, ScalarContainer&& scalars) {
    return MSM(std::begin(std::forward<BaseContainer>(bases)),
               std::end(std::forward<BaseContainer>(bases)),
               std::begin(std::forward<ScalarContainer>(scalars)),
               std::end(std::forward<ScalarContainer>(scalars)));
  }

 private:
  template <typename T>
  FRIEND_TEST(VariableBaseMSMTest, DoMSM);

  template <typename BaseInputIterator, typename ScalarInputIterator>
  static std::vector<ReturnTy> CreateWindowSumsForMSMWindowNAF(
      BaseInputIterator bases_first, ScalarInputIterator scalars_first,
      ScalarInputIterator scalars_last, size_t c) {
    size_t num_bits = ScalarField::kModulusBits;
    size_t digits_count = (num_bits + c - 1) / c;

    std::vector<std::vector<int64_t>> scalar_digits = base::Map(
        scalars_first, scalars_last, [c, num_bits](const ScalarField& scalar) {
          return MakeDigits(scalar.ToBigInt(), c, num_bits);
        });
    std::vector<ReturnTy> window_sums;
    window_sums.reserve(digits_count);
    // TODO(chokobole): Optimize with openmp.
    for (size_t i = 0; i < digits_count; ++i) {
      std::vector<ReturnTy> buckets =
          base::CreateVector(1 << (c - 1), ReturnTy::Zero());
      auto bases_it = bases_first;
      for (size_t j = 0; j < scalar_digits.size(); ++j, ++bases_it) {
        const PointTy& base = *bases_it;
        int64_t scalar = scalar_digits[j][i];
        if (0 < scalar) {
          buckets[static_cast<uint64_t>(scalar - 1)] += base;
        } else if (0 > scalar) {
          buckets[static_cast<uint64_t>(-scalar - 1)] -= base;
        }
      }

      ReturnTy running_sum = ReturnTy::Zero();
      ReturnTy ret = ReturnTy::Zero();
      for (const auto& bucket : base::Reversed(buckets)) {
        running_sum += bucket;
        ret += running_sum;
      }
      window_sums.push_back(std::move(ret));
    }

    return window_sums;
  }

  template <typename BaseInputIterator, typename ScalarInputIterator>
  static std::vector<ReturnTy> CreateWindowSumsForMSM(
      BaseInputIterator bases_first, ScalarInputIterator scalars_first,
      size_t size, size_t c) {
    size_t num_bits = ScalarField::kModulusBits;
    std::vector<size_t> window_starts =
        base::CreateRangedVector(static_cast<size_t>(0), num_bits, c);

    std::function<ReturnTy(size_t)> op = [&bases_first, &scalars_first, size,
                                          c](size_t w_start) {
      ReturnTy ret = ReturnTy::Zero();
      // We don't need the "zero" bucket, so we only have 2^c - 1
      // buckets.
      std::vector<ReturnTy> buckets =
          base::CreateVector((1 << c) - 1, ReturnTy::Zero());
      // This clone is cheap, because the iterator contains just a
      // pointer and an index into the original vectors.
      auto bases_it = bases_first;
      auto scalars_it = scalars_first;
      for (size_t i = 0; i < size; ++i, ++bases_it, ++scalars_it) {
        const ScalarField& scalar = *scalars_it;
        if (scalar.IsZero()) continue;

        auto scalar_bigint = scalar.ToBigInt();
        const PointTy& base = *bases_it;
        if (scalar_bigint.IsOne()) {
          // We only process unit scalars once in the first window.
          if (w_start == 0) {
            ret += base;
          }
        } else {
          // We right-shift by w_start, thus getting rid of the lower
          // bits.
          scalar_bigint.DivBy2ExpInPlace(w_start);

          // We mod the remaining bits by 2^{window size}, thus taking
          // `c` bits.
          uint64_t idx = scalar_bigint[0] % (1 << c);

          // If the scalar is non-zero, we update the corresponding
          // bucket.
          // (Recall that `buckets` doesn't have a zero bucket.)
          if (idx != 0) {
            buckets[idx - 1] += base;
          }
        }
      }

      // Compute sum_{i in 0..num_buckets} and
      // (sum_{j in i..num_buckets} bucket[j])
      // This is computed below for b buckets, using 2b curve additions.
      //
      // We could first normalize `buckets` and then use mixed-addition
      // here, but that's slower for the kinds of groups we care about
      // (Short Weierstrass curves and Twisted Edwards curves).
      // In the case of Short Weierstrass curves,
      // mixed addition saves ~4 field multiplications per addition.
      // However normalization (with the inversion batched) takes ~6
      // field multiplications per element,
      // hence batch normalization is a slowdown.

      // `running_sum` = sum_{j in i..num_buckets} bucket[j],
      // where we iterate backward from i = num_buckets to 0.
      ReturnTy running_sum = ReturnTy::Zero();
      for (const auto& bucket : base::Reversed(buckets)) {
        running_sum += bucket;
        ret += running_sum;
      }
      return ret;
    };

    // TODO(chokobole): Optimize with openmp.
    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    return base::Map(window_starts.begin(), window_starts.end(), std::move(op));
  }

  template <typename BaseInputIterator, typename ScalarInputIterator,
            std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                         PointTy, ScalarField>>* = nullptr>
  static ReturnTy DoMSM(BaseInputIterator bases_first,
                        BaseInputIterator bases_last,
                        ScalarInputIterator scalars_first,
                        ScalarInputIterator scalars_last,
                        bool use_msm_window_naf) {
    size_t size = std::min(std::distance(bases_first, bases_last),
                           std::distance(scalars_first, scalars_last));

    size_t c;
    if (size < 32) {
      c = 3;
    } else {
      c = LnWithoutFloats(size) + 2;
    }

    std::vector<ReturnTy> window_sums;
    if (use_msm_window_naf) {
      window_sums = CreateWindowSumsForMSMWindowNAF(std::move(bases_first),
                                                    std::move(scalars_first),
                                                    std::move(scalars_last), c);
    } else {
      window_sums = CreateWindowSumsForMSM(std::move(bases_first),
                                           std::move(scalars_first), size, c);
    }

    // We store the sum for the lowest window.
    ReturnTy lowest = std::move(*window_sums.begin());
    auto view = absl::MakeConstSpan(window_sums);
    view.remove_prefix(1);

    // We're traversing windows from high to low.
    return lowest + std::accumulate(view.rbegin(), view.rend(),
                                    ReturnTy::Zero(),
                                    [c](ReturnTy& total, const ReturnTy& sum) {
                                      total += sum;
                                      for (size_t i = 0; i < c; ++i) {
                                        total.DoubleInPlace();
                                      }
                                      return total;
                                    });
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_
