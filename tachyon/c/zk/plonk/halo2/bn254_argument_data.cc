#include "tachyon/c/zk/plonk/halo2/bn254_argument_data.h"

#include <utility>

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/plonk/halo2/argument_data.h"

using namespace tachyon;

using Poly =
    math::UnivariateDensePolynomial<math::bn254::Fr, c::math::kMaxDegree>;
using Evals = math::UnivariateEvaluations<math::bn254::Fr, c::math::kMaxDegree>;
using Data = zk::plonk::halo2::ArgumentData<Poly, Evals>;

tachyon_halo2_bn254_argument_data* tachyon_halo2_bn254_argument_data_create(
    size_t num_circuits) {
  return reinterpret_cast<tachyon_halo2_bn254_argument_data*>(
      new Data(num_circuits));
}

void tachyon_halo2_bn254_argument_data_destroy(
    tachyon_halo2_bn254_argument_data* data) {
  delete reinterpret_cast<Data*>(data);
}

void tachyon_halo2_bn254_argument_data_reserve_advice_columns(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_columns) {
  reinterpret_cast<Data*>(data)->advice_columns_vec()[circuit_idx].reserve(
      num_columns);
}

void tachyon_halo2_bn254_argument_data_add_advice_column(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_evaluations* column) {
  reinterpret_cast<Data*>(data)->advice_columns_vec()[circuit_idx].push_back(
      std::move(reinterpret_cast<Evals&>(*column)));
  tachyon_bn254_univariate_evaluations_destroy(column);
}

void tachyon_halo2_bn254_argument_data_reserve_advice_blinds(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_blinds) {
  reinterpret_cast<Data*>(data)->advice_blinds_vec()[circuit_idx].reserve(
      num_blinds);
}

void tachyon_halo2_bn254_argument_data_add_advice_blind(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    const tachyon_bn254_fr* value) {
  reinterpret_cast<Data*>(data)->advice_blinds_vec()[circuit_idx].push_back(
      reinterpret_cast<const math::bn254::Fr&>(*value));
}

void tachyon_halo2_bn254_argument_data_reserve_instance_columns(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_columns) {
  reinterpret_cast<Data*>(data)->instance_columns_vec()[circuit_idx].reserve(
      num_columns);
}

void tachyon_halo2_bn254_argument_data_add_instance_column(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_evaluations* column) {
  reinterpret_cast<Data*>(data)->instance_columns_vec()[circuit_idx].push_back(
      std::move(reinterpret_cast<Evals&>(*column)));
  tachyon_bn254_univariate_evaluations_destroy(column);
}

void tachyon_halo2_bn254_argument_data_reserve_instance_polys(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_polys) {
  reinterpret_cast<Data*>(data)->instance_polys_vec()[circuit_idx].reserve(
      num_polys);
}

void tachyon_halo2_bn254_argument_data_add_instance_poly(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_dense_polynomial* poly) {
  reinterpret_cast<Data*>(data)->instance_polys_vec()[circuit_idx].push_back(
      std::move(reinterpret_cast<Poly&>(*poly)));
  tachyon_bn254_univariate_dense_polynomial_destroy(poly);
}

void tachyon_halo2_bn254_argument_data_reserve_challenges(
    tachyon_halo2_bn254_argument_data* data, size_t num_challenges) {
  reinterpret_cast<Data*>(data)->challenges().reserve(num_challenges);
}

void tachyon_halo2_bn254_argument_data_add_challenge(
    tachyon_halo2_bn254_argument_data* data, const tachyon_bn254_fr* value) {
  reinterpret_cast<Data*>(data)->challenges().push_back(
      reinterpret_cast<const math::bn254::Fr&>(*value));
}
