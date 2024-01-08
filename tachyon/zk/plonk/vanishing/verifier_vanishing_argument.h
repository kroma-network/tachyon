// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VERIFIER_VANISHING_ARGUMENT_H_
#define TACHYON_ZK_PLONK_VANISHING_VERIFIER_VANISHING_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/ref.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/math/elliptic_curves/semigroups.h"
#include "tachyon/zk/base/entities/entity_ty.h"
#include "tachyon/zk/base/verifier_query.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/vanishing/vanishing_committed.h"
#include "tachyon/zk/plonk/vanishing/vanishing_constructed.h"
#include "tachyon/zk/plonk/vanishing/vanishing_evaluated.h"
#include "tachyon/zk/plonk/vanishing/vanishing_partially_evaluated.h"

namespace tachyon::zk {

template <typename PCS, typename Commitment>
[[nodiscard]] bool ReadCommitmentsBeforeY(
    crypto::TranscriptReader<Commitment>* transcript,
    VanishingCommitted<EntityTy::kVerifier, PCS>* committed_out) {
  Commitment c;
  if (!transcript->ReadFromProof(&c)) return false;

  *committed_out = VanishingCommitted<EntityTy::kVerifier, PCS>(std::move(c));
  return true;
}

template <typename PCS, typename Commitment>
[[nodiscard]] bool ReadCommitmentsAfterY(
    VanishingCommitted<EntityTy::kVerifier, PCS>&& committed,
    const VerifyingKey<PCS>& vk,
    crypto::TranscriptReader<Commitment>* transcript,
    VanishingConstructed<EntityTy::kVerifier, PCS>* constructed_out) {
  // Obtain a commitment to h(X) in the form of multiple pieces of degree
  // n - 1
  std::vector<Commitment> h_commitments;
  size_t quotient_poly_degree = vk.constraint_system().ComputeDegree() - 1;
  h_commitments.resize(quotient_poly_degree);
  for (Commitment& commitment : h_commitments) {
    if (!transcript->ReadFromProof(&commitment)) return false;
  }

  *constructed_out = {std::move(h_commitments),
                      std::move(committed).TakeRandomPolyCommitment()};
  return true;
}

template <typename F, typename PCS, typename Commitment>
[[nodiscard]] bool EvaluateAfterX(
    VanishingConstructed<EntityTy::kVerifier, PCS>&& constructed,
    crypto::TranscriptReader<Commitment>* transcript,
    VanishingPartiallyEvaluated<PCS>* partially_evaluated_out) {
  F random_eval;
  if (!transcript->ReadFromProof(&random_eval)) return false;

  *partially_evaluated_out = {std::move(constructed).TakeHCommitments(),
                              std::move(constructed).TakeRandomPolyCommitment(),
                              std::move(random_eval)};
  return true;
}

template <typename PCS, typename Evals, typename F>
VanishingEvaluated<EntityTy::kVerifier, PCS> VerifyVanishingArgument(
    VanishingPartiallyEvaluated<PCS>&& partially_evaluated,
    const Evals& expressions, const F& y, const F& x_n) {
  using Commitment = typename PCS::Commitment;

  F expected_h_eval = F::template LinearCombination</*forward=*/true>(
      expressions.evaluations(), y);
  expected_h_eval /= x_n - F::One();

  // TODO(chokobole): Remove |ToAffine()| since this assumes commitment is an
  // elliptic curve point.
  Commitment h_commitment =
      Commitment::template LinearCombination</*forward=*/false>(
          partially_evaluated.h_commitments(), x_n)
          .ToAffine();

  return {std::move(h_commitment),
          std::move(partially_evaluated).TakeRandomPolyCommitment(),
          std::move(expected_h_eval),
          std::move(partially_evaluated).TakeRandomEval()};
}

template <typename PCS, typename F>
std::vector<VerifierQuery<PCS>> QueryVanishingArgument(
    VanishingEvaluated<EntityTy::kVerifier, PCS>&& evaluated, const F& x) {
  using Commitment = typename PCS::Commitment;

  return {{x, base::Ref<const Commitment>(&evaluated.h_commitment()),
           std::move(evaluated).TakeExpectedHEval()},
          {x, base::Ref<const Commitment>(&evaluated.random_poly_commitment()),
           std::move(evaluated).TakeRandomEval()}};
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VERIFIER_VANISHING_ARGUMENT_H_
