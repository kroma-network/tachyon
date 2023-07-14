#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_GLV_H_

#include <array>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/base/gmp_util.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"

namespace tachyon {
namespace math {

template <typename GLVConfig>
class GLV {
 public:
  using JacobianPointTy = JacobianPoint<typename GLVConfig::Config>;
  using ScalarField = typename JacobianPointTy::ScalarField;

  struct Coefficient {
    bool sign;
    mpz_class value;
  };

  using Coefficients = std::array<GLV<GLVConfig>::Coefficient, 4>;

  struct CoefficientDecompositionResult {
    Coefficient k1;
    Coefficient k2;
  };

  // Decomposes a scalar |k| into k1, k2, s.t. k = k1 + lambda k2,
  static CoefficientDecompositionResult Decompose(const ScalarField& k) {
    std::vector<mpz_class> coefficients =
        base::Map(GLVConfig::ScalarDecompositionCoefficients(),
                  [](const Coefficient& coeff) {
                    return coeff.sign ? coeff.value : -coeff.value;
                  });

    const mpz_class scalar = k.ToMpzClass();
    const mpz_class& n11 = coefficients[0];
    const mpz_class& n12 = coefficients[1];
    const mpz_class& n21 = coefficients[2];
    const mpz_class& n22 = coefficients[3];
    const mpz_class r = ScalarField::Config::Modulus().ToMpzClass();

    // beta = vector([k, 0]) * self.curve.N_inv
    // The inverse of N is 1 / r * Matrix([[n22, -n12], [-n21, n11]]).
    // so β = (k * n22, -k * n12) / r

    mpz_class beta_1 = scalar * n22 / r;
    mpz_class beta_2 = scalar * n12 / r;

    // b = vector([int(beta[0]), int(beta[1])]) * self.curve.N
    // b = (β1 * N11 + β2 * N21, β1 * N12 + β2 * N22) with the signs!
    //   = (b11      + b12     , b21      + b22)   with the signs!

    // b1
    mpz_class b11 = beta_1 * n11;
    mpz_class b12 = beta_2 * n21;
    mpz_class b1 = b11 + b12;

    // b2
    mpz_class b21 = beta_1 * n12;
    mpz_class b22 = beta_2 * n22;
    mpz_class b2 = b21 + b22;

    // k1
    mpz_class k1 = scalar - b1;
    mpz_class k1_abs = gmp::Abs(k1);

    // k2
    mpz_class k2 = -b2;
    mpz_class k2_abs = gmp::Abs(k2);

    return {
        {gmp::IsPositive(k1), std::move(k1_abs)},
        {gmp::IsPositive(k2), std::move(k2_abs)},
    };
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

    if (!result.k1.sign) {
      b1.NegativeInPlace();
    }
    if (!result.k2.sign) {
      b2.NegativeInPlace();
    }

    JacobianPointTy b1b2 = b1 + b2;

    auto k1_begin = gmp::BitIteratorBE::begin(&result.k1.value);
    auto k1_end = gmp::BitIteratorBE::end(&result.k1.value);
    auto k2_begin = gmp::BitIteratorBE::begin(&result.k2.value);

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
