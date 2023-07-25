#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_H_

#include "tachyon/math/elliptic_curves/secp/secp256k1/fq.h"
#include "tachyon/math/elliptic_curves/secp/secp256k1/fr.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon {
namespace math {
namespace secp256k1 {

template <typename Fq, typename Fr>
class CurveConfig {
 public:
  using BaseField = Fq;
  using ScalarField = Fr;

  // clang-format off
  // Parameters are from https://www.secg.org/sec2-v2.pdf#page=13
  // A: Mont(0)
  constexpr static BigInt<4> kA = BigInt<4>(0);
  // B: Mont(7)
  constexpr static BigInt<4> kB = BigInt<4>({
    UINT64_C(30064777911),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
  });
  // X:
  //  Dec: 55066263022277343669578718895168534326250603453777594175500187360389116729240
  //  Hex: 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
  // Y:
  //  Dec: 32670510020758816978083085130507043184471273380659243275938904335757337482424
  //  Hex: 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
  constexpr static Point2<BigInt<4>> kGenerator = Point2<BigInt<4>>(
    BigInt<4>({
      UINT64_C(15507633332195041431),
      UINT64_C(2530505477788034779),
      UINT64_C(10925531211367256732),
      UINT64_C(11061375339145502536),
    }),
    BigInt<4>({
      UINT64_C(12780836216951778274),
      UINT64_C(10231155108014310989),
      UINT64_C(8121878653926228278),
      UINT64_C(14933801261141951190),
    })
  );
  // clang-format on
};

using G1AffinePoint = AffinePoint<SWCurve<CurveConfig<Fq, Fr>>>;
using G1JacobianPoint = JacobianPoint<SWCurve<CurveConfig<Fq, Fr>>>;
#if defined(TACHYON_GMP_BACKEND)
using G1AffinePointGmp = AffinePoint<SWCurve<CurveConfig<FqGmp, FrGmp>>>;
using G1JacobianPointGmp = JacobianPoint<SWCurve<CurveConfig<FqGmp, FrGmp>>>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace secp256k1
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_H_
