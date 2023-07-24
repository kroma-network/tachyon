#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_H_

#include "tachyon/math/elliptic_curves/secp/secp256k1/fq.h"
#include "tachyon/math/elliptic_curves/secp/secp256k1/fr.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"

namespace tachyon {
namespace math {
namespace secp256k1 {

template <typename Fq, typename Fr>
class CurveConfig : public SWCurveConfig<Fq, Fr> {
 public:
  using Config = SWCurveConfig<Fq, Fr>;

  static void Init() {
    Fq::Config::Init();

    Config::B() = Fq(7);

    // clang-format off
    // Parameters are from https://www.secg.org/sec2-v2.pdf#page=13

    // Dec: 55066263022277343669578718895168534326250603453777594175500187360389116729240
    // Hex: 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
    Fq g1_x = Fq::FromDecString("55066263022277343669578718895168534326250603453777594175500187360389116729240");

    // Dec: 32670510020758816978083085130507043184471273380659243275938904335757337482424
    // Hex: 483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
    Fq g1_y = Fq::FromDecString("32670510020758816978083085130507043184471273380659243275938904335757337482424");
    // clang-format on

    Config::Generator() = JacobianPoint<Config>::FromAffine(
        AffinePoint<Config>(Fq(g1_x), Fq(g1_y)));

    Config::Generator() = JacobianPoint<Config>::FromAffine(
        AffinePoint<Config>(std::move(g1_x), std::move(g1_y)));
  }
};

using G1AffinePoint = AffinePoint<CurveConfig<Fq, Fr>::Config>;
using G1JacobianPoint = JacobianPoint<CurveConfig<Fq, Fr>::Config>;
#if defined(TACHYON_GMP_BACKEND)
using G1AffinePointGmp = AffinePoint<CurveConfig<FqGmp, FrGmp>::Config>;
using G1JacobianPointGmp = JacobianPoint<CurveConfig<FqGmp, FrGmp>::Config>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace secp256k1
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_H_
