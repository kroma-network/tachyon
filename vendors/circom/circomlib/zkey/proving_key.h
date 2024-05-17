#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_PROVING_KEY_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_PROVING_KEY_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/substitute.h"

#include "circomlib/zkey/verifying_key.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/zk/r1cs/groth16/proving_key.h"

namespace tachyon::circom {

template <typename Curve>
struct ProvingKey {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

  VerifyingKey<Curve> verifying_key;
  std::vector<G1AffinePoint> ic;
  std::vector<G1AffinePoint> a_g1_query;
  std::vector<G1AffinePoint> b_g1_query;
  std::vector<G2AffinePoint> b_g2_query;
  std::vector<G1AffinePoint> c_g1_query;
  std::vector<G1AffinePoint> h_g1_query;

  bool operator==(const ProvingKey& other) const {
    return verifying_key == other.verifying_key && ic == other.ic &&
           a_g1_query == other.a_g1_query && b_g1_query == other.b_g1_query &&
           b_g2_query == other.b_g2_query && c_g1_query == other.c_g1_query &&
           h_g1_query == other.h_g1_query;
  }
  bool operator!=(const ProvingKey& other) const { return !operator==(other); }

  zk::r1cs::groth16::VerifyingKey<Curve> ToNativeVerifyingKey() const {
    return {
        verifying_key.alpha_g1,
        verifying_key.beta_g2,
        verifying_key.gamma_g2,
        verifying_key.delta_g2,
        ic,
    };
  }

  zk::r1cs::groth16::ProvingKey<Curve> ToNativeProvingKey() const {
    return {
        ToNativeVerifyingKey(),
        verifying_key.beta_g1,
        verifying_key.delta_g1,
        a_g1_query,
        b_g1_query,
        b_g2_query,
        h_g1_query,
        c_g1_query,
    };
  }

  // NOTE(chokobole): the fields are represented in montgomery form.
  std::string ToString() const {
    return absl::Substitute(
        "{verifying_key: $0, ic:$1, a_g1_query: $2, b_g1_query: $3, "
        "b_g2_query: $4, c_g1_query: $5, h_g1_query: $6}",
        verifying_key.ToString(), base::ContainerToString(ic),
        base::ContainerToString(a_g1_query),
        base::ContainerToString(b_g1_query),
        base::ContainerToString(b_g2_query),
        base::ContainerToString(c_g1_query),
        base::ContainerToString(h_g1_query));
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_PROVING_KEY_H_
