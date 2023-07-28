#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_

#include "tachyon/base/static_storage.h"
#include "tachyon/math/base/bit_iterator.h"
#include "tachyon/math/base/gmp/bit_traits.h"
#include "tachyon/math/base/gmp/gmp_identities.h"
#include "tachyon/math/base/gmp/signed_value.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"
#include "tachyon/math/elliptic_curves/projective_point.h"
#include "tachyon/math/matrix/matrix.h"

namespace tachyon::math {

template <typename PointTy>
struct GLVTraits {
  using ReturnType = PointTy;
};

template <typename Config>
struct GLVTraits<AffinePoint<Config>> {
  using ReturnType = JacobianPoint<Config>;
};

template <typename Curve>
class GLV {
 public:
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using AffinePointTy = AffinePoint<Curve>;
  using ProjectivePointTy = ProjectivePoint<Curve>;
  using JacobianPointTy = JacobianPoint<Curve>;
  using PointXYZZTy = PointXYZZ<Curve>;

  using Coefficients = Matrix<mpz_class, 2, 2>;

  struct CoefficientDecompositionResult {
    SignedValue<mpz_class> k1;
    SignedValue<mpz_class> k2;
  };

  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(BaseField, EndomorphismCoefficient);
  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(ScalarField, Lambda);
  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(Coefficients,
                                        ScalarDecompositionCoefficients);

  static void Init() {
    EndomorphismCoefficient() =
        BaseField::FromMontgomery(Curve::Config::kEndomorphismCoefficient);
    Lambda() = ScalarField::FromMontgomery(Curve::Config::kLambda);
    Coefficients() = Matrix<mpz_class, 2, 2>(
        ScalarField::FromMontgomery(Curve::Config::kGLVCoeff00).ToMpzClass(),
        ScalarField::FromMontgomery(Curve::Config::kGLVCoeff01).ToMpzClass(),
        ScalarField::FromMontgomery(Curve::Config::kGLVCoeff10).ToMpzClass(),
        ScalarField::FromMontgomery(Curve::Config::kGLVCoeff11).ToMpzClass());
  }

  static AffinePointTy EndomorphismAffine(const AffinePointTy& point) {
    return AffinePointTy::Endomorphism(point);
  }
  static ProjectivePointTy EndomorphismProjective(
      const ProjectivePointTy& point) {
    return ProjectivePointTy::Endomorphism(point);
  }
  static JacobianPointTy EndomorphismJacobian(const JacobianPointTy& point) {
    return JacobianPointTy::Endomorphism(point);
  }
  static PointXYZZTy EndomorphismXYZZ(const PointXYZZTy& point) {
    return PointXYZZTy::Endomorphism(point);
  }

  // Decomposes a scalar |k| into k1, k2, s.t. k = k1 + lambda k2,
  static CoefficientDecompositionResult Decompose(const ScalarField& k) {
    const Coefficients& coefficients = ScalarDecompositionCoefficients();

    decltype(auto) scalar = k.ToMpzClass();
    const mpz_class& n12 = coefficients[1];
    const mpz_class& n22 = coefficients[3];
    decltype(auto) r = ScalarField::Modulus();

    // NOTE(chokobole): We can't calculate using below directly.
    //
    // Matrix<mpz_class, 1, 2>(scalar, mpz_class(0)) * coefficients.Inverse()
    //
    //
    // This is because the result of |coefficients.Inverse()| is a zero matrix.
    // Therefore, we need to perform a similar operation to overcome the integer
    // issue.
    mpz_class beta_1 = scalar * n22 / r;
    mpz_class beta_2 = scalar * (-n12) / r;

    Matrix<mpz_class, 1, 2> b =
        Matrix<mpz_class, 1, 2>(beta_1, beta_2) * coefficients;

    // k1
    mpz_class k1 = scalar - b[0];

    // k2
    mpz_class k2 = -b[1];

    return {SignedValue<mpz_class>(k1), SignedValue<mpz_class>(k2)};
  }

  template <typename PointTy,
            typename ReturnTy = typename GLVTraits<PointTy>::ReturnType>
  static ReturnTy Mul(const PointTy& p, const ScalarField& k) {
    CoefficientDecompositionResult result = Decompose(k);

    PointTy b1 = p;
    PointTy b2;
    if constexpr (std::is_same_v<PointTy, ProjectivePointTy>) {
      b2 = EndomorphismProjective(p);
    } else if constexpr (std::is_same_v<PointTy, JacobianPointTy>) {
      b2 = EndomorphismJacobian(p);
    } else if constexpr (std::is_same_v<PointTy, PointXYZZTy>) {
      b2 = EndomorphismXYZZ(p);
    } else {
      b2 = EndomorphismAffine(p);
    }

    if (result.k1.sign == Sign::kNegative) {
      b1.NegInPlace();
    }
    if (result.k2.sign == Sign::kNegative) {
      b2.NegInPlace();
    }

    ReturnTy b1b2 = b1 + b2;

    auto k1_begin = BitIteratorBE<mpz_class>::begin(&result.k1.abs_value);
    auto k1_end = BitIteratorBE<mpz_class>::end(&result.k1.abs_value);
    auto k2_begin = BitIteratorBE<mpz_class>::begin(&result.k2.abs_value);

    ReturnTy ret = ReturnTy::Zero();
    bool skip_zeros = true;
    auto k1_it = k1_begin;
    auto k2_it = k2_begin;
    while (k1_it != k1_end) {
      if (skip_zeros && !(*k1_it) && !(*k2_it)) {
        skip_zeros = false;
        ++k1_it;
        ++k2_it;
        continue;
      }
      skip_zeros = false;
      ret.DoubleInPlace();
      if ((*k1_it)) {
        if (*(k2_it)) {
          ret += b1b2;
        } else {
          ret += b1;
        }
      } else {
        if (*(k2_it)) {
          ret += b2;
        }
      }
      ++k1_it;
      ++k2_it;
    }
    return ret;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_
