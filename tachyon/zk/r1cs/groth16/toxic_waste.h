// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_TOXIC_WASTE_H_
#define TACHYON_ZK_R1CS_GROTH16_TOXIC_WASTE_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class ToxicWaste {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using F = typename G1Point::ScalarField;

  ToxicWaste(const F& alpha, const F& beta, const F& gamma, const F& delta,
             const F& x, const G1Point& g1_generator,
             const G2Point& g2_generator)
      : alpha_(alpha),
        beta_(beta),
        gamma_(gamma),
        delta_(delta),
        x_(x),
        g1_generator_(g1_generator),
        g2_generator_(g2_generator) {}
  ToxicWaste(F&& alpha, F&& beta, F&& gamma, F&& delta, F&& x,
             G1Point&& g1_generator, G2Point&& g2_generator)
      : alpha_(std::move(alpha)),
        beta_(std::move(beta)),
        gamma_(std::move(gamma)),
        delta_(std::move(delta)),
        x_(std::move(x)),
        g1_generator_(std::move(g1_generator)),
        g2_generator_(std::move(g2_generator)) {}

  static ToxicWaste RandomWithoutX() {
    return {F::Random(), F::Random(),       F::Random(),      F::Random(),
            F::Zero(),   G1Point::Random(), G2Point::Random()};
  }

  const F& alpha() const { return alpha_; }
  const F& beta() const { return beta_; }
  const F& gamma() const { return gamma_; }
  const F& delta() const { return delta_; }
  const F& x() const { return x_; }
  const G1Point& g1_generator() const { return g1_generator_; }
  const G2Point& g2_generator() const { return g2_generator_; }

  template <typename Domain>
  void SampleX(const Domain* domain) {
    x_ = domain->SampleElementOutsideDomain();
  }

  std::string ToString() const {
    return absl::Substitute(
        "{alpha: $0, beta: $1, gamma: $2, delta: $3, x: $4, g1_generator: $5, "
        "g2_generator: $6}",
        alpha_.ToString(), beta_.ToString(), gamma_.ToString(),
        delta_.ToString(), x_.ToString(), g1_generator_.ToString(),
        g2_generator_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute(
        "{alpha: $0, beta: $1, gamma: $2, delta: $3, x: $4, g1_generator: $5, "
        "g2_generator: $6}",
        alpha_.ToHexString(pad_zero), beta_.ToHexString(pad_zero),
        gamma_.ToHexString(pad_zero), delta_.ToHexString(pad_zero),
        x_.ToHexString(pad_zero), g1_generator_.ToHexString(pad_zero),
        g2_generator_.ToHexString(pad_zero));
  }

 private:
  F alpha_;
  F beta_;
  F gamma_;
  F delta_;
  F x_;
  G1Point g1_generator_;
  G2Point g2_generator_;
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_TOXIC_WASTE_H_
