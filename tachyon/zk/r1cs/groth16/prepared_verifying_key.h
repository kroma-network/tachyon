// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_PREPARED_VERIFYING_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_PREPARED_VERIFYING_KEY_H_

#include <string>
#include <utility>

#include "tachyon/zk/r1cs/groth16/verifying_key.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class PreparedVerifyingKey {
 public:
  using G2Prepared = typename Curve::G2Prepared;
  using Fp12 = typename Curve::Fp12;

  PreparedVerifyingKey() = default;
  PreparedVerifyingKey(const VerifyingKey<Curve>& verifying_key,
                       const Fp12& alpha_g1_beta_g2,
                       const G2Prepared& delta_neg_g2,
                       const G2Prepared& gamma_neg_g2)
      : verifying_key_(verifying_key),
        alpha_g1_beta_g2_(alpha_g1_beta_g2),
        delta_neg_g2_(delta_neg_g2),
        gamma_neg_g2_(gamma_neg_g2) {}
  PreparedVerifyingKey(VerifyingKey<Curve>&& verifying_key,
                       Fp12&& alpha_g1_beta_g2, G2Prepared&& delta_neg_g2,
                       G2Prepared&& gamma_neg_g2)
      : verifying_key_(std::move(verifying_key)),
        alpha_g1_beta_g2_(std::move(alpha_g1_beta_g2)),
        delta_neg_g2_(std::move(delta_neg_g2)),
        gamma_neg_g2_(std::move(gamma_neg_g2)) {}

  const VerifyingKey<Curve>& verifying_key() const { return verifying_key_; }
  const Fp12& alpha_g1_beta_g2() const { return alpha_g1_beta_g2_; }
  const G2Prepared& delta_neg_g2() const { return delta_neg_g2_; }
  const G2Prepared& gamma_neg_g2() const { return gamma_neg_g2_; }

  std::string ToString() const {
    return absl::Substitute(
        "{verifying_key: $0, alpha_g1_beta_g2: $1, delta_neg_g2: $2, "
        "gamma_neg_g2: $3}",
        verifying_key_.ToString(), alpha_g1_beta_g2_.ToString(),
        delta_neg_g2_.ToString(), gamma_neg_g2_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute(
        "{verifying_key: $0, alpha_g1_beta_g2: $1, delta_neg_g2: $2, "
        "gamma_neg_g2: $3}",
        verifying_key_.ToHexString(pad_zero),
        alpha_g1_beta_g2_.ToHexString(pad_zero),
        delta_neg_g2_.ToHexString(pad_zero),
        gamma_neg_g2_.ToHexString(pad_zero));
  }

 private:
  VerifyingKey<Curve> verifying_key_;
  // e([α]₁, [β]₂)
  Fp12 alpha_g1_beta_g2_;
  // [-δ]₂
  G2Prepared delta_neg_g2_;
  // [-γ]₂
  G2Prepared gamma_neg_g2_;
};

template <typename Curve>
PreparedVerifyingKey<Curve> VerifyingKey<Curve>::ToPreparedVerifyingKey() && {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  using G2Prepared = typename Curve::G2Prepared;
  using Fp12 = typename Curve::Fp12;

  G1AffinePoint left[] = {*alpha_g1_};
  G2AffinePoint right[] = {*beta_g2_};
  Fp12 alpha_g1_beta_g2 = math::Pairing<Curve>(left, right);
  G2Prepared delta_neg_g2 = G2Prepared::From(-(*delta_g2_));
  G2Prepared gamma_neg_g2 = G2Prepared::From(-(*gamma_g2_));

  return {std::move(*this), std::move(alpha_g1_beta_g2),
          std::move(delta_neg_g2), std::move(gamma_neg_g2)};
}

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_PREPARED_VERIFYING_KEY_H_
