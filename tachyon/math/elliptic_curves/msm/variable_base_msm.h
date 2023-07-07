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
#include "tachyon/export.h"
#include "tachyon/math/base/gmp_util.h"
#include "tachyon/math/elliptic_curves/msm/msm_util.h"

namespace tachyon {
namespace math {

TACHYON_EXPORT std::vector<int64_t> MakeDigits(const mpz_class& a, size_t w,
                                               size_t num_bits);

template <typename _JacobianPoint>
class VariableBaseMSM {
 public:
  using JacobianPoint = _JacobianPoint;
  using ScalarField = typename JacobianPoint::ScalarField;

  template <
      typename BaseInputIterator, typename ScalarInputIterator,
      std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                   JacobianPoint, ScalarField>>* = nullptr>
  static JacobianPoint MSM(BaseInputIterator bases_first,
                           BaseInputIterator bases_last,
                           ScalarInputIterator scalars_first,
                           ScalarInputIterator scalars_last) {
    return DoMSM(std::move(bases_first), std::move(bases_last),
                 std::move(scalars_first), std::move(scalars_last),
                 JacobianPoint::NEGATION_IS_CHEAP);
  }

 private:
  FRIEND_TEST(VariableBaseMSMTest, DoMSM);

  template <typename BaseInputIterator, typename ScalarInputIterator>
  static std::vector<JacobianPoint> CreateWindowSumsForMSMWindowNAF(
      BaseInputIterator bases_first, ScalarInputIterator scalars_first,
      ScalarInputIterator scalars_last, size_t c) {
    size_t num_bits = ScalarField::MODULUS_BITS;
    size_t digits_count = (num_bits + c - 1) / c;

    std::function<std::vector<int64_t>(const ScalarField&)> op =
        [c, num_bits](const ScalarField& scalar) {
          return MakeDigits(scalar.ToMpzClass(), c, num_bits);
        };

    std::vector<std::vector<int64_t>> scalar_digits =
        base::Map(scalars_first, scalars_last, std::move(op));
    std::vector<JacobianPoint> window_sums;
    window_sums.reserve(digits_count);
    // TODO(chokobole): Optimize with openmp.
    for (size_t i = 0; i < digits_count; ++i) {
      std::vector<JacobianPoint> buckets =
          base::CreateVector(1 << c, JacobianPoint::Zero());
      auto bases_it = bases_first;
      for (size_t j = 0; j < scalar_digits.size(); ++j, ++bases_it) {
        const JacobianPoint& base = *bases_it;
        int64_t scalar = scalar_digits[j][i];
        if (0 < scalar) {
          buckets[static_cast<uint64_t>(scalar - 1)] += base;
        } else if (0 > scalar) {
          buckets[static_cast<uint64_t>(-scalar - 1)] -= base;
        }
      }

      JacobianPoint running_sum = JacobianPoint::Zero();
      JacobianPoint ret = JacobianPoint::Zero();
      for (const auto& bucket : base::Reversed(buckets)) {
        running_sum += bucket;
        ret += running_sum;
      }
      window_sums.push_back(std::move(ret));
    }

    return window_sums;
  }

  template <typename BaseInputIterator, typename ScalarInputIterator>
  static std::vector<JacobianPoint> CreateWindowSumsForMSM(
      BaseInputIterator bases_first, ScalarInputIterator scalars_first,
      size_t size, size_t c) {
    size_t num_bits = ScalarField::MODULUS_BITS;
    std::vector<size_t> window_starts =
        base::CreateRangedVector(static_cast<size_t>(0), num_bits, c);

    std::function<JacobianPoint(size_t)> op = [&bases_first, &scalars_first,
                                               size, c](size_t w_start) {
      JacobianPoint ret = JacobianPoint::Zero();
      // We don't need the "zero" bucket, so we only have 2^c - 1
      // buckets.
      std::vector<JacobianPoint> buckets =
          base::CreateVector((1 << c) - 1, JacobianPoint::Zero());
      // This clone is cheap, because the iterator contains just a
      // pointer and an index into the original vectors.
      auto bases_it = bases_first;
      auto scalars_it = scalars_first;
      for (size_t i = 0; i < size; ++i, ++bases_it, ++scalars_it) {
        ScalarField& scalar = *scalars_it;
        if (scalar.IsZero()) continue;

        const JacobianPoint& base = *bases_it;
        if (scalar.IsOne()) {
          // We only process unit scalars once in the first window.
          if (w_start == 0) {
            ret += base;
          }
        } else {
          // We right-shift by w_start, thus getting rid of the lower
          // bits.
          mpz_class q = scalar.DivBy2Exp(w_start);

          // We mod the remaining bits by 2^{window size}, thus taking
          // `c` bits.
          uint64_t idx = gmp::GetLimb(q, 0) % (1 << c);

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
      JacobianPoint running_sum = JacobianPoint::Zero();
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

  template <
      typename BaseInputIterator, typename ScalarInputIterator,
      std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                   JacobianPoint, ScalarField>>* = nullptr>
  static JacobianPoint DoMSM(BaseInputIterator bases_first,
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

    std::vector<JacobianPoint> window_sums;
    if (use_msm_window_naf) {
      window_sums = CreateWindowSumsForMSMWindowNAF(std::move(bases_first),
                                                    std::move(scalars_first),
                                                    std::move(scalars_last), c);
    } else {
      window_sums = CreateWindowSumsForMSM(std::move(bases_first),
                                           std::move(scalars_first), size, c);
    }

    // We store the sum for the lowest window.
    JacobianPoint lowest = std::move(*window_sums.begin());
    auto view = absl::MakeConstSpan(window_sums);
    view.remove_prefix(1);

    // We're traversing windows from high to low.
    return lowest +
           std::accumulate(view.rbegin(), view.rend(), JacobianPoint::Zero(),
                           [c](JacobianPoint& total, const JacobianPoint& sum) {
                             total += sum;
                             for (size_t i = 0; i < c; ++i) {
                               total.DoubleInPlace();
                             }
                             return total;
                           });
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_
