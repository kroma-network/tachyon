#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_TYPE_TRAITS_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/c/zk/plonk/halo2/bn254_argument_data.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/plonk/halo2/argument_data.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::zk::plonk::halo2::ArgumentData<
    tachyon::math::UnivariateDensePolynomial<tachyon::math::bn254::Fr,
                                             c::math::kMaxDegree>,
    tachyon::math::UnivariateEvaluations<tachyon::math::bn254::Fr,
                                         c::math::kMaxDegree>>> {
  using CType = tachyon_halo2_bn254_argument_data;
};

template <>
struct TypeTraits<tachyon_halo2_bn254_argument_data> {
  using NativeType = tachyon::zk::plonk::halo2::ArgumentData<
      tachyon::math::UnivariateDensePolynomial<tachyon::math::bn254::Fr,
                                               c::math::kMaxDegree>,
      tachyon::math::UnivariateEvaluations<tachyon::math::bn254::Fr,
                                           c::math::kMaxDegree>>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_TYPE_TRAITS_H_
