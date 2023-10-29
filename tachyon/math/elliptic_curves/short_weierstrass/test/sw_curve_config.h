#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_CURVE_CONFIG_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon::math {
namespace test {

template <typename _BaseField, typename _ScalarField>
class SWCurveConfig {
 public:
  using BaseField = _BaseField;
  using BasePrimeField = BaseField;
  using ScalarField = _ScalarField;

  using CpuBaseField = typename BaseField::CpuField;
  using CpuScalarField = typename ScalarField::CpuField;
  using GpuBaseField = typename BaseField::GpuField;
  using GpuScalarField = typename ScalarField::GpuField;
  using CpuCurveConfig = SWCurveConfig<CpuBaseField, CpuScalarField>;
  using GpuCurveConfig = SWCurveConfig<GpuBaseField, GpuScalarField>;

  constexpr static bool kAIsZero = true;

  static BaseField kA;
  static BaseField kB;
  static Point2<BaseField> kGenerator;

  static void Init() {
    kA = BaseField::Zero();
    kB = BaseField(5);
    kGenerator.x = BaseField(5);
    kGenerator.y = BaseField(5);
  }
};

template <typename BaseField, typename ScalarField>
BaseField SWCurveConfig<BaseField, ScalarField>::kA;
template <typename BaseField, typename ScalarField>
BaseField SWCurveConfig<BaseField, ScalarField>::kB;
template <typename BaseField, typename ScalarField>
Point2<BaseField> SWCurveConfig<BaseField, ScalarField>::kGenerator;

using G1Curve = SWCurve<SWCurveConfig<GF7, GF7>>;
using AffinePoint = math::AffinePoint<G1Curve>;
using ProjectivePoint = math::ProjectivePoint<G1Curve>;
using JacobianPoint = math::JacobianPoint<G1Curve>;
using PointXYZZ = math::PointXYZZ<G1Curve>;
#if defined(TACHYON_GMP_BACKEND)
using G1CurveGmp = SWCurve<SWCurveConfig<GF7Gmp, GF7Gmp>>;
using AffinePointGmp = math::AffinePoint<G1CurveGmp>;
using ProjectivePointGmp = math::ProjectivePoint<G1CurveGmp>;
using JacobianPointGmp = math::JacobianPoint<G1CurveGmp>;
using PointXYZZGmp = math::PointXYZZ<G1CurveGmp>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace test
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_CURVE_CONFIG_H_
