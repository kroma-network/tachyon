// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/unpermuted_table.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::zk {

namespace {

class UnpermutedTableTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = 7;

  using PCS =
      crypto::KZGCommitmentScheme<math::bn254::G1AffinePoint,
                                  math::bn254::G2AffinePoint, kMaxDegree,
                                  math::bn254::G1AffinePoint>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(UnpermutedTableTest, Construct) {
  constexpr size_t kMaxDegree = 7;
  constexpr size_t kCols = 4;

  using F = math::bn254::G1Curve::ScalarField;

  std::unique_ptr<math::UnivariateEvaluationDomain<F, kMaxDegree>> domain =
      math::UnivariateEvaluationDomain<F, kMaxDegree>::Create(kMaxDegree + 1);
  UnpermutedTable<PCS> unpermuted_table =
      UnpermutedTable<PCS>::Construct(kCols, domain.get());
  F omega = domain->group_gen();
  std::vector<F> omega_powers = domain->GetRootsOfUnity(kMaxDegree + 1, omega);

  F delta = unpermuted_table.GetDelta();
  EXPECT_NE(delta, F::One());
  EXPECT_EQ(delta.Pow(F::Config::kTrace), F::One());
  for (size_t i = 1; i < kCols; ++i) {
    for (size_t j = 0; j < kMaxDegree + 1; ++j) {
      omega_powers[j] *= delta;
      EXPECT_EQ(omega_powers[j], unpermuted_table[Label(i, j)]);
    }
  }
}
}  // namespace tachyon::zk
