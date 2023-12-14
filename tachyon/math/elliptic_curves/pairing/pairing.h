#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_PAIRING_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_PAIRING_H_

namespace tachyon::math {

template <typename Curve, typename G1AffinePointContainer,
          typename G2PreparedContainer>
auto Pairing(const G1AffinePointContainer& a, const G2PreparedContainer& b) {
  return Curve::FinalExponentiation(Curve::MultiMillerLoop(a, b));
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_PAIRING_H_
