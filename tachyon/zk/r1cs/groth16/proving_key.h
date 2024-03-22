// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_PROVING_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_PROVING_KEY_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "tachyon/zk/r1cs/groth16/verifying_key.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class ProvingKey : public Key {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using F = typename G1Point::ScalarField;

  const VerifyingKey<Curve>& verifying_key() const { return verifying_key_; }
  VerifyingKey<Curve>&& TakeVerifyingKey() && {
    return std::move(verifying_key_);
  }

  const G1Point& beta_g1() const { return beta_g1_; }
  const G1Point& delta_g1() const { return delta_g1_; }
  const std::vector<G1Point>& a_g1_query() const { return a_g1_query_; }
  const std::vector<G1Point>& b_g1_query() const { return b_g1_query_; }
  const std::vector<G2Point>& b_g2_query() const { return b_g2_query_; }
  const std::vector<G1Point>& h_g1_query() const { return h_g1_query_; }
  const std::vector<G1Point>& l_g1_query() const { return l_g1_query_; }

  template <size_t MaxDegree, typename QAP>
  [[nodiscard]] bool Load(ToxicWaste<Curve>& toxic_waste,
                          const Circuit<F>& circuit) {
    using G1JacobianPoint = typename Curve::G1Curve::JacobianPoint;
    using G2JacobianPoint = typename Curve::G2Curve::JacobianPoint;

    F gamma_delta_inverse[] = {
        toxic_waste.gamma(),
        toxic_waste.delta(),
    };
    CHECK(F::BatchInverseInPlace(gamma_delta_inverse));

    KeyPreLoadResult<G1Point, MaxDegree> result;
    PreLoad<QAP>(toxic_waste, circuit, &result);

    QAPInstanceMapResult<F>& qap_instance_map_result =
        result.qap_instance_map_result;
    math::FixedBaseMSM<G1Point>& g1_msm = result.g1_msm;

    if (!verifying_key_.Load(toxic_waste, result, gamma_delta_inverse[0]))
      return false;

    const F& delta_inverse = gamma_delta_inverse[1];

    // |h[i]| = (xⁱ * t(x)) / δ
    std::vector<F> h =
        QAP::ComputeHQuery(result.domain.get(), qap_instance_map_result.t_x,
                           toxic_waste.x(), delta_inverse);

    // |l[i]| = (β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / δ
    std::vector<F> l(qap_instance_map_result.num_witness_variables);
    size_t num_instance_variables =
        qap_instance_map_result.num_instance_variables;
    std::vector<F>& a = qap_instance_map_result.a;
    std::vector<F>& b = qap_instance_map_result.b;
    std::vector<F>& c = qap_instance_map_result.c;
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l.size(); ++i) {
      l[i] = ComputeABC(
          a[num_instance_variables + i], b[num_instance_variables + i],
          c[num_instance_variables + i], toxic_waste, delta_inverse);
    }

    math::FixedBaseMSM<G2Point> g2_msm;
    g2_msm.Reset(result.non_zero_b, toxic_waste.g2_generator());

    std::vector<G1JacobianPoint> a_g1_query_jacobian(
        qap_instance_map_result.a.size());
    if (!g1_msm.Run(a, &a_g1_query_jacobian)) return false;
    a.clear();

    std::vector<G1JacobianPoint> b_g1_query_jacobian(b.size());
    if (!g1_msm.Run(b, &b_g1_query_jacobian)) return false;

    std::vector<G2JacobianPoint> b_g2_query_jacobian(b.size());
    if (!g2_msm.Run(b, &b_g2_query_jacobian)) return false;
    b.clear();

    std::vector<G1JacobianPoint> h_g1_query_jacobian(h.size());
    if (!g1_msm.Run(h, &h_g1_query_jacobian)) return false;
    h.clear();

    std::vector<G1JacobianPoint> l_g1_query_jacobian(l.size());
    if (!g1_msm.Run(l, &l_g1_query_jacobian)) return false;
    l.clear();

    G1JacobianPoint beta_delta_jacobian[] = {
        toxic_waste.beta() * toxic_waste.g1_generator(),
        toxic_waste.delta() * toxic_waste.g1_generator(),
    };
    G1Point beta_delta[2];
    if (!G1JacobianPoint::BatchNormalize(beta_delta_jacobian, &beta_delta))
      return false;

    beta_g1_ = std::move(beta_delta[0]);
    delta_g1_ = std::move(beta_delta[1]);

    a_g1_query_.resize(a_g1_query_jacobian.size());
    if (!G1JacobianPoint::BatchNormalize(a_g1_query_jacobian, &a_g1_query_))
      return false;
    b_g1_query_.resize(b_g1_query_jacobian.size());
    if (!G1JacobianPoint::BatchNormalize(b_g1_query_jacobian, &b_g1_query_))
      return false;
    b_g2_query_.resize(b_g2_query_jacobian.size());
    if (!G2JacobianPoint::BatchNormalize(b_g2_query_jacobian, &b_g2_query_))
      return false;
    h_g1_query_.resize(h_g1_query_jacobian.size());
    if (!G1JacobianPoint::BatchNormalize(h_g1_query_jacobian, &h_g1_query_))
      return false;
    l_g1_query_.resize(l_g1_query_jacobian.size());
    return G1JacobianPoint::BatchNormalize(l_g1_query_jacobian, &l_g1_query_);
  }

 private:
  VerifyingKey<Curve> verifying_key_;
  // [β]₁
  G1Point beta_g1_;
  // [δ]₁
  G1Point delta_g1_;
  // |a_g1_query_[i]| = [aᵢ(x)]₁
  std::vector<G1Point> a_g1_query_;
  // |b_g1_query_[i]| = [bᵢ(x)]₁
  std::vector<G1Point> b_g1_query_;
  // |b_g2_query_[i]| = [bᵢ(x)]₂
  std::vector<G2Point> b_g2_query_;
  // |h_g1_query_[i]| = [(xⁱ * t(x)) / δ]₁
  std::vector<G1Point> h_g1_query_;
  // |l_g1_query_[i]| = [(β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / δ]₁
  std::vector<G1Point> l_g1_query_;
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_PROVING_KEY_H_
