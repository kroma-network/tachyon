#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_PAIRING_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_PAIRING_H_

#include "tachyon/base/containers/container_util.h"

namespace tachyon::math {

template <typename Curve, typename G1AffinePointContainer,
          typename G2AffinePointContainer>
auto Pairing(const G1AffinePointContainer& a, const G2AffinePointContainer& b) {
  using G2Prepared = typename Curve::G2Prepared;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  return Curve::FinalExponentiation(
      Curve::MultiMillerLoop(a, base::Map(b, [](const G2AffinePoint& point) {
                               return G2Prepared::From(point);
                             })));
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_PAIRING_H_
