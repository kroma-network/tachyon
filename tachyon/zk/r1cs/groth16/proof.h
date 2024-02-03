// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_PROOF_H_
#define TACHYON_ZK_R1CS_GROTH16_PROOF_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class Proof {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;

  Proof(const G1Point& a, const G2Point& b, const G1Point& c)
      : a_(a), b_(b), c_(c) {}
  Proof(G1Point&& a, G2Point&& b, G1Point&& c)
      : a_(std::move(a)), b_(std::move(b)), c_(std::move(c)) {}

  const G1Point& a() const { return a_; }
  const G2Point& b() const { return b_; }
  const G1Point& c() const { return c_; }

  bool operator==(const Proof& other) const {
    return a_ == other.a_ && b_ == other.b_ && c_ == other.c_;
  }
  bool operator!=(const Proof& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("{a: $0, b: $1, c: $2}", a_.ToString(),
                            b_.ToString(), c_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("{a: $0, b: $1, c: $2}", a_.ToHexString(pad_zero),
                            b_.ToHexString(pad_zero), c_.ToHexString(pad_zero));
  }

 private:
  G1Point a_;
  G2Point b_;
  G1Point c_;
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_PROOF_H_
