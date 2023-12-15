// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "gtest/gtest.h"

#include "tachyon/zk/base/entities/verifier_base.h"
#include "tachyon/zk/plonk/circuit/examples/simple_circuit.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"
#include "tachyon/zk/plonk/vanishing/prover_vanishing_argument.h"
#include "tachyon/zk/plonk/vanishing/verifier_vanishing_argument.h"

namespace tachyon::zk {

namespace {

class VanishingArgumentTest : public halo2::ProverTest {};

}  // namespace

TEST_F(VanishingArgumentTest, VanishingArgument) {
  VanishingCommitted<EntityTy::kProver, PCS> committed_p;
  ASSERT_TRUE(CommitRandomPoly(prover_.get(), &committed_p));

  SimpleCircuit<F> circuit = SimpleCircuit<F>();
  VerifyingKey<PCS> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  ExtendedEvals extended_evals =
      ExtendedEvals::One(prover_->extended_domain()->size() - 1);
  VanishingConstructed<EntityTy::kProver, PCS> constructed_p;
  ASSERT_TRUE(CommitFinalHPoly(prover_.get(), std::move(committed_p), vkey,
                               extended_evals, &constructed_p));

  F x = F::One();
  VanishingEvaluated<EntityTy::kProver, PCS> evaluated;
  ASSERT_TRUE(CommitRandomEval(prover_->pcs(), std::move(constructed_p), x,
                               F::One(), prover_->GetWriter(), &evaluated));

  std::vector<ProverQuery<PCS>> h_x =
      OpenVanishingArgument(std::move(evaluated), x);

  base::Buffer read_buf(prover_->GetWriter()->buffer().buffer(),
                        prover_->GetWriter()->buffer().buffer_len());
  std::unique_ptr<crypto::TranscriptReader<Commitment>> reader =
      absl::WrapUnique(
          new halo2::Blake2bReader<Commitment>(std::move(read_buf)));

  std::unique_ptr<VerifierBase<PCS>> verifier =
      std::make_unique<VerifierBase<PCS>>(
          VerifierBase<PCS>(prover_->TakePCS(), std::move(reader)));
  verifier->set_domain(prover_->TakeDomain());
  verifier->set_extended_domain(prover_->TakeExtendedDomain());

  VanishingCommitted<EntityTy::kVerifier, PCS> committed_v;
  ASSERT_TRUE(ReadCommitmentsBeforeY(verifier->GetReader(), &committed_v));

  VanishingConstructed<EntityTy::kVerifier, PCS> constructed_v;
  ASSERT_TRUE(ReadCommitmentsAfterY(std::move(committed_v), vkey,
                                    verifier->GetReader(), &constructed_v));

  VanishingPartiallyEvaluated<PCS> partially_evaluated_v;
  ASSERT_TRUE(EvaluateAfterX<F>(std::move(constructed_v), verifier->GetReader(),
                                &partially_evaluated_v));

  F y = F::One();
  Evals evals = Evals::One(kMaxDegree);
  VanishingEvaluated<EntityTy::kVerifier, PCS> evaluated_v =
      VerifyVanishingArgument(std::move(partially_evaluated_v), evals, y, F(2));

  QueryVanishingArgument(std::move(evaluated_v), x);
}

}  // namespace tachyon::zk
