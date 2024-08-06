// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_VERIFY_H_
#define TACHYON_ZK_R1CS_GROTH16_VERIFY_H_

#include "absl/types/span.h"

#include "tachyon/base/profiler.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/pairing/pairing.h"
#include "tachyon/math/geometry/point_conversions.h"
#include "tachyon/zk/r1cs/groth16/prepared_verifying_key.h"
#include "tachyon/zk/r1cs/groth16/proof.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve, typename Container, typename Bucket>
[[nodiscard]] bool PrepareInputs(const PreparedVerifyingKey<Curve>& pvk,
                                 const Container& public_inputs,
                                 Bucket* prepared_inputs) {
  using G1Point = typename Curve::G1Curve::AffinePoint;

  TRACE_EVENT("ProofVerification", "Groth16::PrepareInputs");

  absl::Span<const G1Point> l_g1_query = pvk.verifying_key().l_g1_query();
  absl::Span<const G1Point> l_g1_query_first_skipped = l_g1_query.subspan(1);
  math::VariableBaseMSM<G1Point> msm;
  if (!msm.Run(l_g1_query_first_skipped, public_inputs, prepared_inputs))
    return false;
  *prepared_inputs += l_g1_query[0];
  return true;
}

template <typename Curve, typename Bucket>
[[nodiscard]] bool VerifyProofWithPreparedInputs(
    const PreparedVerifyingKey<Curve>& pvk, const Proof<Curve>& proof,
    const Bucket& prepared_inputs) {
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Prepared = typename Curve::G2Prepared;

  TRACE_EVENT("ProofVerification", "Groth16::VerifyProofWithPreparedInputs");

  // clang-format off
  // e(A, B) * e([Σ xᵢ * (γ⁻¹ * (β * aᵢ(x) + α * bᵢ(x) + cᵢ(x))]₁, [-γ]₂) * e(C, [-δ]₂) ≟ e([α]₁, [β]₂)
  // clang-format on
  G1Point g1[] = {
      proof.a(),
      math::ConvertPoint<G1Point>(prepared_inputs),
      proof.c(),
  };
  G2Prepared g2[] = {
      G2Prepared::From(proof.b()),
      pvk.gamma_neg_g2(),
      pvk.delta_neg_g2(),
  };
  return math::Pairing<Curve>(g1, g2) == pvk.alpha_g1_beta_g2();
}

template <typename Curve, typename Container>
[[nodiscard]] bool VerifyProof(const PreparedVerifyingKey<Curve>& pvk,
                               const Proof<Curve>& proof,
                               const Container& public_inputs) {
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using Bucket = typename math::VariableBaseMSM<G1Point>::Bucket;

  TRACE_EVENT("ProofVerification", "Groth16::VerifyProof");

  Bucket prepared_inputs;
  if (!PrepareInputs(pvk, public_inputs, &prepared_inputs)) return false;
  return VerifyProofWithPreparedInputs(pvk, proof, prepared_inputs);
}

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_VERIFY_H_
