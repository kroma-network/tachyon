#include "tachyon/c/zk/plonk/halo2/bn254_argument_data.h"

#include <utility>

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_argument_data_type_traits.h"

using namespace tachyon;

using Poly =
    math::UnivariateDensePolynomial<math::bn254::Fr, c::math::kMaxDegree>;
using Evals = math::UnivariateEvaluations<math::bn254::Fr, c::math::kMaxDegree>;
using Data = zk::plonk::halo2::ArgumentData<Poly, Evals>;

tachyon_halo2_bn254_argument_data* tachyon_halo2_bn254_argument_data_create(
    size_t num_circuits) {
  return c::base::c_cast(new Data(num_circuits));
}

void tachyon_halo2_bn254_argument_data_destroy(
    tachyon_halo2_bn254_argument_data* data) {
  delete c::base::native_cast(data);
}

void tachyon_halo2_bn254_argument_data_reserve_advice_columns(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_columns) {
  c::base::native_cast(data)->advice_columns_vec()[circuit_idx].reserve(
      num_columns);
}

void tachyon_halo2_bn254_argument_data_add_advice_column(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_evaluations* column) {
  c::base::native_cast(data)->advice_columns_vec()[circuit_idx].push_back(
      std::move(c::base::native_cast(*column)));
  tachyon_bn254_univariate_evaluations_destroy(column);
}

void tachyon_halo2_bn254_argument_data_reserve_advice_blinds(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_blinds) {
  c::base::native_cast(data)->advice_blinds_vec()[circuit_idx].reserve(
      num_blinds);
}

void tachyon_halo2_bn254_argument_data_add_advice_blind(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    const tachyon_bn254_fr* value) {
  c::base::native_cast(data)->advice_blinds_vec()[circuit_idx].push_back(
      reinterpret_cast<const math::bn254::Fr&>(*value));
}

void tachyon_halo2_bn254_argument_data_reserve_instance_columns(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_columns) {
  c::base::native_cast(data)->instance_columns_vec()[circuit_idx].reserve(
      num_columns);
}

void tachyon_halo2_bn254_argument_data_add_instance_column(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_evaluations* column) {
  c::base::native_cast(data)->instance_columns_vec()[circuit_idx].push_back(
      std::move(c::base::native_cast(*column)));
  tachyon_bn254_univariate_evaluations_destroy(column);
}

void tachyon_halo2_bn254_argument_data_reserve_instance_polys(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_polys) {
  c::base::native_cast(data)->instance_polys_vec()[circuit_idx].reserve(
      num_polys);
}

void tachyon_halo2_bn254_argument_data_add_instance_poly(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_dense_polynomial* poly) {
  c::base::native_cast(data)->instance_polys_vec()[circuit_idx].push_back(
      std::move(c::base::native_cast(*poly)));
  tachyon_bn254_univariate_dense_polynomial_destroy(poly);
}

void tachyon_halo2_bn254_argument_data_reserve_challenges(
    tachyon_halo2_bn254_argument_data* data, size_t num_challenges) {
  c::base::native_cast(data)->challenges().reserve(num_challenges);
}

void tachyon_halo2_bn254_argument_data_add_challenge(
    tachyon_halo2_bn254_argument_data* data, const tachyon_bn254_fr* value) {
  c::base::native_cast(data)->challenges().push_back(
      tachyon::c::base::native_cast(*value));
}
