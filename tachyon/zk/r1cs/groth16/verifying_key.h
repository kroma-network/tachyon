// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_VERIFYING_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_VERIFYING_KEY_H_

#include "absl/types/span.h"

#include "tachyon/zk/r1cs/groth16/key.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class PreparedVerifyingKey;

template <typename Curve>
class VerifyingKey : public Key {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using F = typename G1Point::ScalarField;

  VerifyingKey() = default;
  VerifyingKey(const G1Point* alpha_g1, const G2Point* beta_g2,
               const G2Point* gamma_g2, const G2Point* delta_g2,
               absl::Span<const G1Point> l_g1_query)
      : alpha_g1_(alpha_g1),
        beta_g2_(beta_g2),
        gamma_g2_(gamma_g2),
        delta_g2_(delta_g2),
        l_g1_query_(l_g1_query) {}

  const G1Point& alpha_g1() const { return *alpha_g1_; }
  const G2Point& beta_g2() const { return *beta_g2_; }
  const G2Point& gamma_g2() const { return *gamma_g2_; }
  const G2Point& delta_g2() const { return *delta_g2_; }
  absl::Span<const G1Point> l_g1_query() const { return l_g1_query_; }

  PreparedVerifyingKey<Curve> ToPreparedVerifyingKey() const;

 protected:
  // [α]₁
  const G1Point* alpha_g1_ = nullptr;
  // [β]₂
  const G2Point* beta_g2_ = nullptr;
  // [γ]₂
  const G2Point* gamma_g2_ = nullptr;
  // [δ]₂
  const G2Point* delta_g2_ = nullptr;
  // |l_g1_query_[i]| = [(β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / γ]₁
  absl::Span<const G1Point> l_g1_query_;
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_VERIFYING_KEY_H_
