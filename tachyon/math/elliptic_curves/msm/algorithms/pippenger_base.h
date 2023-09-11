#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_

#include "absl/types/span.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"

namespace tachyon::math {

template <typename PointTy>
class PippengerTraits {
 public:
  using Bucket = typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;
};

template <typename Curve>
class PippengerTraits<AffinePoint<Curve>> {
 public:
  using Bucket = PointXYZZ<Curve>;
};

template <typename PointTy,
          typename Bucket_ = typename PippengerTraits<PointTy>::Bucket>
class PippengerBase {
 public:
  using Bucket = Bucket_;

  static Bucket AccumulateBuckets(
      absl::Span<const Bucket> buckets,
      const Bucket& initial_value = Bucket::Zero()) {
    Bucket running_sum = Bucket::Zero();
    Bucket window_sum = initial_value;

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

  static Bucket AccumulateWindowSums(absl::Span<const Bucket> window_sums,
                                     size_t window_bits) {
    // We store the sum for the lowest window.
    Bucket lowest = window_sums.front();
    window_sums.remove_prefix(1);

    // We're traversing windows from high to low.
    return lowest +
           std::accumulate(window_sums.rbegin(), window_sums.rend(),
                           Bucket::Zero(),
                           [window_bits](Bucket& total, const Bucket& sum) {
                             total += sum;
                             for (size_t i = 0; i < window_bits; ++i) {
                               total.DoubleInPlace();
                             }
                             return total;
                           });
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_
