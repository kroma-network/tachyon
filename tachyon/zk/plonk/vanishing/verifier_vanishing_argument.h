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

template <typename PCSTy, typename Commitment>
[[nodiscard]] bool ReadCommitmentsBeforeY(
    crypto::TranscriptReader<Commitment>* transcript,
    VanishingCommitted<EntityTy::kVerifier, PCSTy>* committed_out) {
  Commitment c;
  if (!transcript->ReadPoint(&c)) return false;

  *committed_out = VanishingCommitted<EntityTy::kVerifier, PCSTy>(std::move(c));
  return true;
}

template <typename PCSTy, typename Commitment>
[[nodiscard]] bool ReadCommitmentsAfterY(
    VanishingCommitted<EntityTy::kVerifier, PCSTy>&& committed,
    const VerifyingKey<PCSTy>& vk,
    crypto::TranscriptReader<Commitment>* transcript,
    VanishingConstructed<EntityTy::kVerifier, PCSTy>* constructed_out) {
  // Obtain a commitment to h(X) in the form of multiple pieces of degree
  // n - 1
  std::vector<Commitment> h_commitments;
  size_t quotient_poly_degree = vk.constraint_system().ComputeDegree() - 1;
  h_commitments.resize(quotient_poly_degree);
  for (Commitment& commitment : h_commitments) {
    if (!transcript->ReadPoint(&commitment)) return false;
  }

  *constructed_out = {std::move(h_commitments),
                      std::move(committed).TakeRandomPolyCommitment()};
  return true;
}

template <typename F, typename PCSTy, typename Commitment>
[[nodiscard]] bool EvaluateAfterX(
    VanishingConstructed<EntityTy::kVerifier, PCSTy>&& constructed,
    crypto::TranscriptReader<Commitment>* transcript,
    VanishingPartiallyEvaluated<PCSTy>* partially_evaluated_out) {
  F random_eval;
  if (!transcript->ReadScalar(&random_eval)) return false;

  *partially_evaluated_out = {std::move(constructed).TakeHCommitments(),
                              std::move(constructed).TakeRandomPolyCommitment(),
                              std::move(random_eval)};
  return true;
}

template <typename PCSTy, typename Evals, typename F>
VanishingEvaluated<EntityTy::kVerifier, PCSTy> VerifyVanishingArgument(
    VanishingPartiallyEvaluated<PCSTy>&& partially_evaluated,
    const Evals& expressions, const crypto::Challenge255<F>& y, const F& x_n) {
  using Commitment = typename PCSTy::Commitment;
  using AdditiveResultTy =
      typename math::internal::AdditiveSemigroupTraits<Commitment>::ReturnTy;

  F y_scalar = y.ChallengeAsScalar();
  F expected_h_eval = std::accumulate(
      expressions.evaluations().begin(), expressions.evaluations().end(),
      F::Zero(), [y_scalar](F& h_eval, const F& v) {
        h_eval *= y_scalar;
        return h_eval += v;
      });
  expected_h_eval /= x_n - F::One();

  Commitment h_commitment =
      std::accumulate(
          partially_evaluated.h_commitments().rbegin(),
          partially_evaluated.h_commitments().rend(), AdditiveResultTy::Zero(),
          [&x_n](AdditiveResultTy& acc, const Commitment& commitment) {
            acc *= x_n;
            return acc += commitment;
          })
          .ToAffine();

  return {std::move(h_commitment),
          std::move(partially_evaluated).TakeRandomPolyCommitment(),
          std::move(expected_h_eval),
          std::move(partially_evaluated).TakeRandomEval()};
}

template <typename PCSTy, typename F>
std::vector<VerifierQuery<PCSTy>> QueryVanishingArgument(
    VanishingEvaluated<EntityTy::kVerifier, PCSTy>&& evaluated,
    const crypto::Challenge255<F>& x) {
  using Commitment = typename PCSTy::Commitment;

  F x_scalar = x.ChallengeAsScalar();

  return {{x_scalar, base::Ref<const Commitment>(&evaluated.h_commitment()),
           std::move(evaluated).TakeExpectedHEval()},
          {std::move(x_scalar),
           base::Ref<const Commitment>(&evaluated.random_poly_commitment()),
           std::move(evaluated).TakeRandomEval()}};
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VERIFIER_VANISHING_ARGUMENT_H_
