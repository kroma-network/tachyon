// clang-format off
#include "absl/base/call_once.h"

#include "tachyon/base/logging.h"
#include "%{base_field_hdr}"
#include "%{scalar_field_hdr}"
%{if HasGLVCoefficients}
#include "tachyon/math/base/gmp/gmp_util.h"
%{endif HasGLVCoefficients}
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve.h"

namespace %{namespace} {

template <typename _BaseField, typename _ScalarField>
class %{class}CurveConfig {
 public:
  using BaseField = _BaseField;
  using BasePrimeField = %{base_prime_field};
  using ScalarField = _ScalarField;

  using CpuBaseField = typename BaseField::CpuField;
  using CpuScalarField = typename ScalarField::CpuField;
  using GpuBaseField = typename BaseField::GpuField;
  using GpuScalarField = typename ScalarField::GpuField;
  using CpuCurveConfig = %{class}CurveConfig<CpuBaseField, CpuScalarField>;
  using GpuCurveConfig = %{class}CurveConfig<GpuBaseField, GpuScalarField>;

  constexpr static bool kAIsZero = %{a_is_zero};

  // TODO(chokobole): Make them constexpr.
  static BaseField kA;
  static BaseField kB;
  static Point2<BaseField> kGenerator;
%{if HasGLVCoefficients}
  static BaseField kEndomorphismCoefficient;
%{endif HasGLVCoefficients}
  static ScalarField kLambda;
  static mpz_class kGLVCoeffs[4];

  static void Init() {
    static absl::once_flag once;
    absl::call_once(once, &%{class}CurveConfig::DoInit);
  }

  constexpr static BaseField MulByA(const BaseField& v) {
%{mul_by_a_code}
  }

 private:
  static void DoInit() {
%{a_init}
%{b_init}
%{x_init}
%{y_init}
%{if HasGLVCoefficients}
%{endomorphism_coefficient_init_code}
%{endif HasGLVCoefficients}
%{lambda_init_code}
%{glv_coeffs_init_code}
    VLOG(1) << "%{namespace}::%{class} initialized";
  }
};

template <typename BaseField, typename ScalarField>
BaseField %{class}CurveConfig<BaseField, ScalarField>::kA;
template <typename BaseField, typename ScalarField>
BaseField %{class}CurveConfig<BaseField, ScalarField>::kB;
template <typename BaseField, typename ScalarField>
Point2<BaseField> %{class}CurveConfig<BaseField, ScalarField>::kGenerator;
%{if HasGLVCoefficients}
template <typename BaseField, typename ScalarField>
BaseField %{class}CurveConfig<BaseField, ScalarField>::kEndomorphismCoefficient;
%{endif HasGLVCoefficients}
template <typename BaseField, typename ScalarField>
ScalarField %{class}CurveConfig<BaseField, ScalarField>::kLambda;
template <typename BaseField, typename ScalarField>
mpz_class %{class}CurveConfig<BaseField, ScalarField>::kGLVCoeffs[4];

using %{class}Curve = SWCurve<%{class}CurveConfig<%{base_field}, %{scalar_field}>>;
using %{class}AffinePoint = AffinePoint<%{class}Curve>;
using %{class}ProjectivePoint = ProjectivePoint<%{class}Curve>;
using %{class}JacobianPoint = JacobianPoint<%{class}Curve>;
using %{class}PointXYZZ = PointXYZZ<%{class}Curve>;

}  // namespace %{namespace}
// clang-format on
