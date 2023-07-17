#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_

#include "tachyon/math/base/gmp/bit_iterator.h"
#include "tachyon/math/base/gmp/gmp_identities.h"
#include "tachyon/math/base/gmp/signed_value.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/matrix/matrix.h"

namespace tachyon {
namespace math {

template <typename GLVConfig>
class GLV {
 public:
  using JacobianPointTy = JacobianPoint<typename GLVConfig::Config>;
  using ScalarField = typename JacobianPointTy::ScalarField;

  using Coefficients = Matrix<mpz_class, 2, 2>;

  struct CoefficientDecompositionResult {
    SignedValue<mpz_class> k1;
    SignedValue<mpz_class> k2;
  };

  // Decomposes a scalar |k| into k1, k2, s.t. k = k1 + lambda k2,
  static CoefficientDecompositionResult Decompose(const ScalarField& k) {
    const Coefficients& coefficients =
        GLVConfig::ScalarDecompositionCoefficients();

    decltype(auto) scalar = k.ToMpzClass();
    const mpz_class& n12 = coefficients[1];
    const mpz_class& n22 = coefficients[3];
    decltype(auto) r = ScalarField::Config::Modulus().ToMpzClass();

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

  template <typename Point>
  static JacobianPointTy Mul(const Point& p, const ScalarField& k) {
    CoefficientDecompositionResult result = Decompose(k);

    Point b1 = p;
    Point b2;
    if constexpr (std::is_same_v<Point, JacobianPointTy>) {
      b2 = GLVConfig::Endomorphism(p);
    } else {
      b2 = GLVConfig::EndomorphismAffine(p);
    }

    if (result.k1.sign == Sign::kNegative) {
      b1.NegInPlace();
    }
    if (result.k2.sign == Sign::kNegative) {
      b2.NegInPlace();
    }

    JacobianPointTy b1b2 = b1 + b2;

    auto k1_begin = BitIteratorBE<mpz_class>::begin(&result.k1.abs_value);
    auto k1_end = BitIteratorBE<mpz_class>::end(&result.k1.abs_value);
    auto k2_begin = BitIteratorBE<mpz_class>::begin(&result.k2.abs_value);

    JacobianPointTy ret = JacobianPointTy::Zero();
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

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_
