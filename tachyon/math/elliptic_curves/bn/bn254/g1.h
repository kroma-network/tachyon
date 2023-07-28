#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"

namespace tachyon::math {
namespace bn254 {

template <typename Fq, typename Fr>
class G1CurveConfig {
 public:
  using BaseField = Fq;
  using ScalarField = Fr;

  // clang-format off
  // A: Mont(0)
  constexpr static BigInt<4> kA = BigInt<4>(0);
  // B: Mont(3)
  constexpr static BigInt<4> kB = BigInt<4>({
    UINT64_C(8797723225643362519),
    UINT64_C(2263834496217719225),
    UINT64_C(3696305541684646532),
    UINT64_C(3035258219084094862),
  });
  // Generator: (Mont(1), Mont(2))
  constexpr static Point2<BigInt<4>> kGenerator = Point2<BigInt<4>>(
    BigInt<4>({
      UINT64_C(15230403791020821917),
      UINT64_C(754611498739239741),
      UINT64_C(7381016538464732716),
      UINT64_C(1011752739694698287),
    }),
    BigInt<4>({
      UINT64_C(12014063508332092218),
      UINT64_C(1509222997478479483),
      UINT64_C(14762033076929465432),
      UINT64_C(2023505479389396574),
    })
  );
  // clang-format on
};

using G1AffinePoint = AffinePoint<SWCurve<G1CurveConfig<Fq, Fr>>>;
using G1ProjectivePoint = ProjectivePoint<SWCurve<G1CurveConfig<Fq, Fr>>>;
using G1JacobianPoint = JacobianPoint<SWCurve<G1CurveConfig<Fq, Fr>>>;
using G1PointXYZZ = PointXYZZ<SWCurve<G1CurveConfig<Fq, Fr>>>;
#if defined(TACHYON_GMP_BACKEND)
using G1AffinePointGmp = AffinePoint<SWCurve<G1CurveConfig<FqGmp, FrGmp>>>;
using G1ProjectivePointGmp =
    ProjectivePoint<SWCurve<G1CurveConfig<FqGmp, FrGmp>>>;
using G1JacobianPointGmp = JacobianPoint<SWCurve<G1CurveConfig<FqGmp, FrGmp>>>;
using G1PointXYZZGmp = PointXYZZ<SWCurve<G1CurveConfig<FqGmp, FrGmp>>>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace bn254
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_
