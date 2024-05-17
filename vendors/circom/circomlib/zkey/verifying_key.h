#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_VERIFYING_KEY_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_VERIFYING_KEY_H_

#include <stdint.h>

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"

namespace tachyon {
namespace circom {

template <typename Curve>
struct VerifyingKey {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

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

  // NOTE(chokobole): the fields are represented in montgomery form.
  std::string ToString() const {
    return absl::Substitute(
        "{alpha_g1: $0, beta_g1: $1, beta_g2: $2, gamma_g2: $3, delta_g1: $4, "
        "delta_g2: $5}",
        alpha_g1.ToString(), beta_g1.ToString(), beta_g2.ToString(),
        gamma_g2.ToString(), delta_g1.ToString(), delta_g2.ToString());
  }
};

}  // namespace circom

namespace base {

template <typename Curve>
class Copyable<circom::VerifyingKey<Curve>> {
 public:
  static bool WriteTo(const circom::VerifyingKey<Curve>& vk, Buffer* buffer) {
    NOTREACHED();
    return false;
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       circom::VerifyingKey<Curve>* vk) {
    using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
    using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

    G1AffinePoint alpha_g1;
    G1AffinePoint beta_g1;
    G2AffinePoint beta_g2;
    G2AffinePoint gamma_g2;
    G1AffinePoint delta_g1;
    G2AffinePoint delta_g2;
    if (!buffer.ReadMany(&alpha_g1, &beta_g1, &beta_g2, &gamma_g2, &delta_g1,
                         &delta_g2))
      return false;
    *vk = {std::move(alpha_g1), std::move(beta_g1),  std::move(beta_g2),
           std::move(gamma_g2), std::move(delta_g1), std::move(delta_g2)};
    return true;
  }

  static size_t EstimateSize(const circom::VerifyingKey<Curve>& vk) {
    return base::EstimateSize(vk.alpha_g1, vk.beta_g1, vk.beta_g2, vk.gamma_g2,
                              vk.delta_g1, vk.delta_g2);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_VERIFYING_KEY_H_
