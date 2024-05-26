// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_OWNED_PREPARED_VERIFYING_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_OWNED_PREPARED_VERIFYING_KEY_H_

#include <utility>

#include "tachyon/zk/r1cs/groth16/owned_verifying_key.h"
#include "tachyon/zk/r1cs/groth16/prepared_verifying_key.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class OwnedPreparedVerifyingKey : public PreparedVerifyingKey<Curve> {
 public:
  using G2Prepared = typename Curve::G2Prepared;
  using Fp12 = typename Curve::Fp12;

  OwnedPreparedVerifyingKey() = default;
  OwnedPreparedVerifyingKey(const OwnedVerifyingKey<Curve>& owned_verifying_key,
                            const Fp12& alpha_g1_beta_g2,
                            const G2Prepared& delta_neg_g2,
                            const G2Prepared& gamma_neg_g2)
      : PreparedVerifyingKey<Curve>(alpha_g1_beta_beta, delta_neg_g2,
                                    gamma_neg_g2),
        owned_verifying_key_(owned_verifying_key) {
    SetParentValues();
  }
  OwnedPreparedVerifyingKey(OwnedVerifyingKey<Curve>&& owned_verifying_key,
                            Fp12&& alpha_g1_beta_g2, G2Prepared&& delta_neg_g2,
                            G2Prepared&& gamma_neg_g2)
      : PreparedVerifyingKey<Curve>(std::move(alpha_g1_beta_beta),
                                    std::move(delta_neg_g2),
                                    std::move(gamma_neg_g2)),
        owned_verifying_key_(std::move(owned_verifying_key)) {
    SetParentValues();
  }
  OwnedPreparedVerifyingKey(const OwnedPreparedVerifyingKey& other)
      : PreparedVerifyingKey<Curve>(other),
        owned_verifying_key_(other.owned_verifying_key_) {
    SetParentValues();
  }
  OwnedPreparedVerifyingKey& operator=(const OwnedPreparedVerifyingKey& other) {
    PreparedVerifyingKey::operator=(other);
    owned_verifying_key_ = other.owned_verifying_key_;
    SetParentValues();
    return *this;
  }
  OwnedPreparedVerifyingKey(OwnedPreparedVerifyingKey&& other)
      : PreparedVerifyingKey<Curve>(std::move(other)),
        owned_verifying_key_(std::move(other.owned_verifying_key_)) {
    SetParentValues();
  }
  OwnedPreparedVerifyingKey& operator=(OwnedPreparedVerifyingKey&& other) {
    PreparedVerifyingKey::operator=(std::move(other));
    owned_verifying_key_ = std::move(other.owned_verifying_key_);
    SetParentValues();
    return *this;
  }

 private:
  void SetParentValues() { this->verifying_key_ = owned_verifying_key_; }

  OwnedVerifyingKey<Curve> owned_verifying_key_;
};

template <typename Curve>
OwnedPreparedVerifyingKey<Curve>
OwnedVerifyingKey<Curve>::ToOwnedPreparedVerifyingKey() && {
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

#endif  // TACHYON_ZK_R1CS_GROTH16_OWNED_PREPARED_VERIFYING_KEY_H_
