#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_VERIFYING_KEY_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_VERIFYING_KEY_H_

#include <stdint.h>

#include <string>

#include "absl/strings/substitute.h"

#include "circomlib/base/g1_affine_point.h"
#include "circomlib/base/g2_affine_point.h"

namespace tachyon::circom {

struct VerifyingKey {
  G1AffinePoint alpha_g1;
  G1AffinePoint beta_g1;
  G2AffinePoint beta_g2;
  G2AffinePoint gamma_g2;
  G1AffinePoint delta_g1;
  G2AffinePoint delta_g2;

  bool operator==(const VerifyingKey& other) const {
    return alpha_g1 == other.alpha_g1 && beta_g1 == other.beta_g1 &&
           beta_g2 == other.beta_g2 && gamma_g2 == other.gamma_g2 &&
           delta_g1 == other.delta_g1 && delta_g2 == other.delta_g2;
  }
  bool operator!=(const VerifyingKey& other) const {
    return !operator==(other);
  }

  bool Read(const base::ReadOnlyBuffer& buffer, uint32_t field_size) {
    return alpha_g1.Read(buffer, field_size) &&
           beta_g1.Read(buffer, field_size) &&
           beta_g2.Read(buffer, field_size) &&
           gamma_g2.Read(buffer, field_size) &&
           delta_g1.Read(buffer, field_size) &&
           delta_g2.Read(buffer, field_size);
  }

  // NOTE(chokobole): the fields are represented in montgomery form.
  std::string ToString() const {
    return absl::Substitute(
        "{alpha_g1: $0, beta_g1: $1, beta_g2: $2, gamma_g2: $3, delta_g1: $4, "
        "delta_g2: $5}",
        alpha_g1.ToString(), beta_g1.ToString(), beta_g2.ToString(),
        gamma_g2.ToString(), delta_g1.ToString(), delta_g2.ToString());
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_VERIFYING_KEY_H_
