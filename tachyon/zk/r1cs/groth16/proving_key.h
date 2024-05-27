// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_PROVING_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_PROVING_KEY_H_

#include "tachyon/zk/r1cs/groth16/verifying_key.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class ProvingKey : public Key {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using F = typename G1Point::ScalarField;

  ProvingKey() = default;
  ProvingKey(const VerifyingKey<Curve>& verifying_key, const G1Point* beta_g1,
             const G1Point* delta_g1, absl::Span<const G1Point> a_g1_query,
             absl::Span<const G1Point> b_g1_query,
             absl::Span<const G2Point> b_g2_query,
             absl::Span<const G1Point> h_g1_query,
             absl::Span<const G1Point> l_g1_query)
      : verifying_key_(verifying_key),
        beta_g1_(beta_g1),
        delta_g1_(delta_g1),
        a_g1_query_(a_g1_query),
        b_g1_query_(b_g1_query),
        b_g2_query_(b_g2_query),
        h_g1_query_(h_g1_query),
        l_g1_query_(l_g1_query) {}

  const VerifyingKey<Curve>& verifying_key() const { return verifying_key_; }

  const G1Point& beta_g1() const { return *beta_g1_; }
  const G1Point& delta_g1() const { return *delta_g1_; }
  absl::Span<const G1Point> a_g1_query() const { return a_g1_query_; }
  absl::Span<const G1Point> b_g1_query() const { return b_g1_query_; }
  absl::Span<const G2Point> b_g2_query() const { return b_g2_query_; }
  absl::Span<const G1Point> h_g1_query() const { return h_g1_query_; }
  absl::Span<const G1Point> l_g1_query() const { return l_g1_query_; }

 protected:
  VerifyingKey<Curve> verifying_key_;
  // [β]₁
  const G1Point* beta_g1_ = nullptr;
  // [δ]₁
  const G1Point* delta_g1_ = nullptr;
  // |a_g1_query_[i]| = [aᵢ(x)]₁
  absl::Span<const G1Point> a_g1_query_;
  // |b_g1_query_[i]| = [bᵢ(x)]₁
  absl::Span<const G1Point> b_g1_query_;
  // |b_g2_query_[i]| = [bᵢ(x)]₂
  absl::Span<const G2Point> b_g2_query_;
  // |h_g1_query_[i]| = [(xⁱ * t(x)) / δ]₁
  absl::Span<const G1Point> h_g1_query_;
  // |l_g1_query_[i]| = [(β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / δ]₁
  absl::Span<const G1Point> l_g1_query_;
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_PROVING_KEY_H_
