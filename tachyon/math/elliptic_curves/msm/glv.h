// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_

#include "tachyon/math/base/bit_iterator.h"
#include "tachyon/math/base/gmp/bit_traits.h"
#include "tachyon/math/base/gmp/signed_value.h"
#include "tachyon/math/elliptic_curves/semigroups.h"
#include "tachyon/math/matrix/gmp_num_traits.h"

namespace tachyon::math {

template <typename Point>
class GLV {
 public:
  using BaseField = typename Point::BaseField;
  using ScalarField = typename Point::ScalarField;
  using RetPoint = typename internal::AdditiveSemigroupTraits<Point>::ReturnTy;

  struct CoefficientDecompositionResult {
    SignedValue<mpz_class> k1;
    SignedValue<mpz_class> k2;
  };

  static Point Endomorphism(const Point& point) {
    return Point::Endomorphism(point);
  }
  // Decomposes a scalar |k| into k1, k2, s.t. k = k1 + lambda k2,
  static CoefficientDecompositionResult Decompose(const ScalarField& k) {
    using Config = typename Point::Curve::Config;

    Eigen::Matrix<mpz_class, 2, 2> coefficients(
        {{Config::kGLVCoeffs[0], Config::kGLVCoeffs[1]},
         {Config::kGLVCoeffs[2], Config::kGLVCoeffs[3]}});

    decltype(auto) scalar = k.ToMpzClass();
    const mpz_class& n12 = coefficients(0, 1);
    const mpz_class& n22 = coefficients(1, 1);
    mpz_class r;
    gmp::WriteLimbs(ScalarField::Config::kModulus.limbs, ScalarField::kLimbNums,
                    &r);

    // clang-format off
    // NOTE(chokobole): We can't calculate using below directly.
    //
    // Eigen::Matrix<mpz_class, 1, 2>(scalar, mpz_class(0)) * coefficients.inverse()
    //
    // Eigen matrix emits an error like below:
    //
    // external/eigen_archive/Eigen/src/LU/InverseImpl.h:352:3: error: static assertion failed: THIS_FUNCTION_IS_NOT_FOR_INTEGER_NUMERIC_TYPES
    // 352 |   EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::IsInteger,THIS_FUNCTION_IS_NOT_FOR_INTEGER_NUMERIC_TYPES)
    // clang-format on
    mpz_class beta_1 = scalar * n22 / r;
    mpz_class beta_2 = scalar * (-n12) / r;

    Eigen::Matrix<mpz_class, 1, 2> b =
        Eigen::Matrix<mpz_class, 1, 2>{beta_1, beta_2} * coefficients;

    // k1
    mpz_class k1 = scalar - b[0];

    // k2
    mpz_class k2 = -b[1];

    return {SignedValue<mpz_class>(k1), SignedValue<mpz_class>(k2)};
  }

  static RetPoint Mul(const Point& p, const ScalarField& k) {
    CoefficientDecompositionResult result = Decompose(k);

    Point b1 = p;
    Point b2 = Endomorphism(p);

    if (result.k1.sign == Sign::kNegative) {
      b1.NegateInPlace();
    }
    if (result.k2.sign == Sign::kNegative) {
      b2.NegateInPlace();
    }

    RetPoint b1b2 = b1 + b2;

    auto k1_begin = BitIteratorBE<mpz_class>::begin(&result.k1.abs_value);
    auto k1_end = BitIteratorBE<mpz_class>::end(&result.k1.abs_value);
    auto k2_begin = BitIteratorBE<mpz_class>::begin(&result.k2.abs_value);

    RetPoint ret = RetPoint::Zero();
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
