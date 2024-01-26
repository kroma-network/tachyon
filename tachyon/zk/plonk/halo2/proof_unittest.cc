// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/halo2/proof.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "tachyon/zk/base/row_index.h"

namespace tachyon::zk::halo2 {
namespace {

using F = math::bn254::Fr;
using Commitment = math::bn254::G1AffinePoint;

template <typename T>
std::vector<std::vector<T>> CreateRandomElementsVec(RowIndex rows,
                                                    size_t cols) {
  return base::CreateVector(
      rows, [cols]() { return base::CreateVector(cols, T::Random()); });
}

std::vector<std::vector<zk::LookupPair<Commitment>>> CreateRandomLookupPairsVec(
    RowIndex rows, size_t cols) {
  return base::CreateVector(rows, [cols]() {
    return base::CreateVector(cols, []() {
      return zk::LookupPair<Commitment>(Commitment::Random(),
                                        Commitment::Random());
    });
  });
}

}  // namespace

TEST(ProofTest, JsonValueConverter) {
  math::bn254::G1Curve::Init();

  size_t num_circuits_ = 2;
  size_t num_elements_ = 3;

  Proof<F, Commitment> expected_proof;
  expected_proof.advices_commitments_vec =
      CreateRandomElementsVec<Commitment>(num_circuits_, num_elements_);
  expected_proof.challenges = base::CreateVector(num_circuits_, F::Random());
  expected_proof.theta = F::Random();
  expected_proof.lookup_permuted_commitments_vec =
      CreateRandomLookupPairsVec(num_circuits_, num_elements_);
  expected_proof.beta = F::Random();
  expected_proof.gamma = F::Random();
  expected_proof.permutation_product_commitments_vec =
      CreateRandomElementsVec<Commitment>(num_circuits_, num_elements_);
  expected_proof.lookup_product_commitments_vec =
      CreateRandomElementsVec<Commitment>(num_circuits_, num_elements_);
  expected_proof.vanishing_random_poly_commitment = Commitment::Random();
  expected_proof.y = F::Random();
  expected_proof.vanishing_h_poly_commitments =
      base::CreateVector(5, Commitment::Random());
  expected_proof.x = F::Random();
  expected_proof.instance_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  expected_proof.advice_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  expected_proof.fixed_evals = base::CreateVector(num_circuits_, F::Random());
  expected_proof.vanishing_random_eval = F::Random();
  expected_proof.common_permutation_evals =
      base::CreateVector(num_circuits_, F::Random());
  expected_proof.permutation_product_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  expected_proof.permutation_product_next_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  expected_proof.permutation_product_last_evals_vec = {
      base::CreateVector(5, std::optional<F>(F::Random())),
      base::CreateVector(5, std::optional<F>())};
  expected_proof.lookup_product_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  expected_proof.lookup_product_next_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  expected_proof.lookup_permuted_input_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  expected_proof.lookup_permuted_input_inv_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  expected_proof.lookup_permuted_table_evals_vec =
      CreateRandomElementsVec<F>(num_circuits_, num_elements_);
  std::string json = base::WriteToJson(expected_proof);

  Proof<F, Commitment> proof;
  std::string error;
  ASSERT_TRUE(base::ParseJson(json, &proof, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(proof, expected_proof);
}

}  // namespace tachyon::zk::halo2
