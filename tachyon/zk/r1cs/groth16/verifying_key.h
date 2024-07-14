// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_VERIFYING_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_VERIFYING_KEY_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/base/optional.h"
#include "tachyon/base/strings/string_util.h"
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
  VerifyingKey(const G1Point& alpha_g1, const G2Point& beta_g2,
               const G2Point& gamma_g2, const G2Point& delta_g2,
               const std::vector<G1Point>& l_g1_query)
      : alpha_g1_(alpha_g1),
        beta_g2_(beta_g2),
        gamma_g2_(gamma_g2),
        delta_g2_(delta_g2),
        l_g1_query_(l_g1_query) {}
  VerifyingKey(G1Point&& alpha_g1, G2Point&& beta_g2, G2Point&& gamma_g2,
               G2Point&& delta_g2, std::vector<G1Point>&& l_g1_query)
      : alpha_g1_(std::move(alpha_g1)),
        beta_g2_(std::move(beta_g2)),
        gamma_g2_(std::move(gamma_g2)),
        delta_g2_(std::move(delta_g2)),
        l_g1_query_(std::move(l_g1_query)) {}

  const G1Point& alpha_g1() const { return alpha_g1_; }
  const G2Point& beta_g2() const { return beta_g2_; }
  const G2Point& gamma_g2() const { return gamma_g2_; }
  const G2Point& delta_g2() const { return delta_g2_; }
  const std::vector<G1Point>& l_g1_query() const { return l_g1_query_; }

  template <size_t MaxDegree, typename QAP>
  [[nodiscard]] bool Load(ToxicWaste<Curve>& toxic_waste,
                          const Circuit<F>& circuit) {
    KeyPreLoadResult<G1Point, MaxDegree> result;
    PreLoad<QAP>(toxic_waste, circuit, &result);
    return Load(toxic_waste, result);
  }

  template <size_t MaxDegree>
  [[nodiscard]] bool Load(const ToxicWaste<Curve>& toxic_waste,
                          KeyPreLoadResult<G1Point, MaxDegree>& result) {
    F gamma_inverse = unwrap(toxic_waste.gamma().Inverse());
    return Load(toxic_waste, result, gamma_inverse);
  }

  template <size_t MaxDegree>
  [[nodiscard]] bool Load(const ToxicWaste<Curve>& toxic_waste,
                          KeyPreLoadResult<G1Point, MaxDegree>& result,
                          const F& gamma_inverse) {
    using G1JacobianPoint = typename Curve::G1Curve::JacobianPoint;
    using G2JacobianPoint = typename Curve::G2Curve::JacobianPoint;

    const QAPInstanceMapResult<F>& qap_instance_map_result =
        result.qap_instance_map_result;
    math::FixedBaseMSM<G1Point>& g1_msm = result.g1_msm;

    // |l[i]| = (β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / γ
    std::vector<F> l(qap_instance_map_result.num_instance_variables);
    const std::vector<F>& a = qap_instance_map_result.a;
    const std::vector<F>& b = qap_instance_map_result.b;
    const std::vector<F>& c = qap_instance_map_result.c;
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l.size(); ++i) {
      l[i] = ComputeABC(a[i], b[i], c[i], toxic_waste, gamma_inverse);
    }

    std::vector<G1JacobianPoint> l_g1_query_jacobian(l.size());
    if (!g1_msm.Run(l, &l_g1_query_jacobian)) return false;
    l.clear();

    alpha_g1_ = (toxic_waste.alpha() * toxic_waste.g1_generator()).ToAffine();

    G2JacobianPoint beta_gamma_delta_jacobian[] = {
        toxic_waste.beta() * toxic_waste.g2_generator(),
        toxic_waste.gamma() * toxic_waste.g2_generator(),
        toxic_waste.delta() * toxic_waste.g2_generator(),
    };
    G2Point beta_gamma_delta[3];
    if (!G2JacobianPoint::BatchNormalize(beta_gamma_delta_jacobian,
                                         &beta_gamma_delta))
      return false;
    beta_g2_ = std::move(beta_gamma_delta[0]);
    gamma_g2_ = std::move(beta_gamma_delta[1]);
    delta_g2_ = std::move(beta_gamma_delta[2]);

    l_g1_query_.resize(l_g1_query_jacobian.size());
    return G1JacobianPoint::BatchNormalize(l_g1_query_jacobian, &l_g1_query_);
  }

  PreparedVerifyingKey<Curve> ToPreparedVerifyingKey() &&;

  std::string ToString() const {
    return absl::Substitute(
        "{alpha_g1: $0, beta_g2: $1, gamma_g2: $2, delta_g2: $3, l_g1_query: "
        "$4}",
        alpha_g1_.ToString(), beta_g2_.ToString(), gamma_g2_.ToString(),
        delta_g2_.ToString(), base::ContainerToString(l_g1_query_));
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute(
        "{alpha_g1: $0, beta_g2: $1, gamma_g2: $2, delta_g2: $3, l_g1_query: "
        "$4}",
        alpha_g1_.ToHexString(pad_zero), beta_g2_.ToHexString(pad_zero),
        gamma_g2_.ToHexString(pad_zero), delta_g2_.ToHexString(pad_zero),
        base::ContainerToHexString(l_g1_query_, pad_zero));
  }

 private:
  // [α]₁
  G1Point alpha_g1_;
  // [β]₂
  G2Point beta_g2_;
  // [γ]₂
  G2Point gamma_g2_;
  // [δ]₂
  G2Point delta_g2_;
  // |l_g1_query_[i]| = [(β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / γ]₁
  std::vector<G1Point> l_g1_query_;
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_VERIFYING_KEY_H_
