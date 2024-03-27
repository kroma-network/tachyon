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

struct ProvingKey {
  VerifyingKey verifying_key;
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

  template <typename Curve>
  zk::r1cs::groth16::VerifyingKey<Curve> ToNativeVerifyingKey() const {
    using G1Curve = typename Curve::G1Curve;
    using G2Curve = typename Curve::G2Curve;

    return {
        verifying_key.alpha_g1.ToNative<true, G1Curve>(),
        verifying_key.beta_g2.ToNative<true, G2Curve>(),
        verifying_key.gamma_g2.ToNative<true, G2Curve>(),
        verifying_key.delta_g2.ToNative<true, G2Curve>(),
        base::Map(ic,
                  [](const G1AffinePoint& point) {
                    return point.ToNative<true, G1Curve>();
                  }),
    };
  }

  template <typename Curve>
  zk::r1cs::groth16::ProvingKey<Curve> ToNativeProvingKey() const {
    using G1Curve = typename Curve::G1Curve;
    using G2Curve = typename Curve::G2Curve;

    return {
        ToNativeVerifyingKey<Curve>(),
        verifying_key.beta_g1.ToNative<true, G1Curve>(),
        verifying_key.delta_g1.ToNative<true, G1Curve>(),
        base::Map(a_g1_query,
                  [](const G1AffinePoint& point) {
                    return point.ToNative<true, G1Curve>();
                  }),
        base::Map(b_g1_query,
                  [](const G1AffinePoint& point) {
                    return point.ToNative<true, G1Curve>();
                  }),
        base::Map(b_g2_query,
                  [](const G2AffinePoint& point) {
                    return point.ToNative<true, G2Curve>();
                  }),
        base::Map(h_g1_query,
                  [](const G1AffinePoint& point) {
                    return point.ToNative<true, G1Curve>();
                  }),
        base::Map(c_g1_query,
                  [](const G1AffinePoint& point) {
                    return point.ToNative<true, G1Curve>();
                  }),
    };
  }

  // NOTE(chokobole): the fields are represented in montgomery form.
  std::string ToString() const {
    return absl::Substitute(
        "{verifying_key: $0, ic:$1, a_g1_query: $2, b_g1_query: $3, "
        "b_g2_query: $4, c_g1_query: $5, h_g1_query: $6}",
        verifying_key.ToString(), base::VectorToString(ic),
        base::VectorToString(a_g1_query), base::VectorToString(b_g1_query),
        base::VectorToString(b_g2_query), base::VectorToString(c_g1_query),
        base::VectorToString(h_g1_query));
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_PROVING_KEY_H_
