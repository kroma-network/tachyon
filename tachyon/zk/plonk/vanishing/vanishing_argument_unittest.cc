// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/vanishing/vanishing_argument.h"

#include "gtest/gtest.h"

#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/verifier_base.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/examples/simple_circuit.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/vanishing/circuit_polynomial_builder.h"
#include "tachyon/zk/plonk/vanishing/prover_vanishing_argument.h"

namespace tachyon::zk::plonk {

namespace {

class VanishingArgumentTest : public halo2::ProverTest {
 public:
  BlindedPolynomial<Poly> GenRandomBlindedPoly() const {
    return {prover_->domain()->Random<Poly>(), F::Random()};
  }
  Poly GenRandomPoly() const { return prover_->domain()->Random<Poly>(); }
};

}  // namespace

TEST_F(VanishingArgumentTest, BuildExtendedCircuitColumn) {
  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, SimpleFloorPlanner> circuit(constant, a, b);

  ProvingKey<Poly, Evals, Commitment> pkey;
  ASSERT_TRUE(pkey.Load(prover_.get(), circuit));

  std::vector<Poly> instance_columns = {GenRandomPoly()};
  std::vector<Poly> advice_columns = {GenRandomPoly(), GenRandomPoly()};
  std::vector<Poly> fixed_columns = {GenRandomPoly(), GenRandomPoly()};
  RefTable<Poly> table(absl::MakeConstSpan(fixed_columns),
                       absl::MakeConstSpan(advice_columns),
                       absl::MakeConstSpan(instance_columns));
  std::vector<RefTable<Poly>> poly_tables = {table};

  std::vector<F> challenges = base::CreateVector(0, F::Random());
  F y = F::Random();
  F beta = F::Random();
  F gamma = F::Random();
  F theta = F::Random();
  F zeta = GetZeta<F>();

  size_t cs_degree = pkey.verifying_key().constraint_system().ComputeDegree();
  std::vector<PermutationCommitted<Poly>> committed_permutations =
      base::CreateVector(1, [this, cs_degree]() {
        std::vector<BlindedPolynomial<Poly>> product_polys =
            base::CreateVector(cs_degree - 2, GenRandomBlindedPoly());
        return PermutationCommitted<Poly>(std::move(product_polys));
      });

  std::vector<std::vector<LookupCommitted<Poly>>> committed_lookups_vec =
      base::CreateVector(1, [this]() {
        return base::CreateVector(0, [this]() {
          return LookupCommitted<Poly>(GenRandomBlindedPoly(),
                                       GenRandomBlindedPoly(),
                                       GenRandomBlindedPoly());
        });
      });

  VanishingArgument<F> vanishing_argument =
      VanishingArgument<F>::Create(pkey.verifying_key().constraint_system());

  ExtendedEvals circuit_column = vanishing_argument.BuildExtendedCircuitColumn(
      prover_.get(), pkey, beta, gamma, theta, y, zeta, challenges,
      committed_permutations, committed_lookups_vec, poly_tables);

  EXPECT_FALSE(circuit_column.IsZero());
}

TEST_F(VanishingArgumentTest, VanishingArgument) {
  VanishingCommitted<Poly> committed_p;
  ASSERT_TRUE(CommitRandomPoly(prover_.get(), &committed_p));

  SimpleCircuit<F, SimpleFloorPlanner> circuit =
      SimpleCircuit<F, SimpleFloorPlanner>();
  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  ExtendedEvals extended_evals =
      ExtendedEvals::One(prover_->extended_domain()->size() - 1);
  VanishingConstructed<Poly> constructed_p;
  ASSERT_TRUE(CommitFinalHPoly(prover_.get(), std::move(committed_p), vkey,
                               extended_evals, &constructed_p));

  F x = F::One();
  VanishingEvaluated<Poly> evaluated;
  ASSERT_TRUE(CommitRandomEval(prover_->pcs(), std::move(constructed_p), x,
                               F::One(), prover_->GetWriter(), &evaluated));
}

}  // namespace tachyon::zk::plonk
