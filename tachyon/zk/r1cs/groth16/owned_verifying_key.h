// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_OWNED_VERIFYING_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_OWNED_VERIFYING_KEY_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/zk/r1cs/groth16/verifying_key.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class OwnedPreparedVerifyingKey;

template <typename Curve>
class OwnedVerifyingKey : public VerifyingKey<Curve> {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using F = typename G1Point::ScalarField;

  OwnedVerifyingKey() = default;
  OwnedVerifyingKey(const G1Point& owned_alpha_g1, const G2Point& owned_beta_g2,
                    const G2Point& owned_gamma_g2,
                    const G2Point& owned_delta_g2,
                    const std::vector<G1Point>& owned_l_g1_query)
      : owned_alpha_g1_(owned_alpha_g1),
        owned_beta_g2_(owned_beta_g2),
        owned_gamma_g2_(owned_gamma_g2),
        owned_delta_g2_(owned_delta_g2),
        owned_l_g1_query_(owned_l_g1_query) {
    SetParentValues();
  }
  OwnedVerifyingKey(G1Point&& owned_alpha_g1, G2Point&& owned_beta_g2,
                    G2Point&& owned_gamma_g2, G2Point&& owned_delta_g2,
                    std::vector<G1Point>&& owned_l_g1_query)
      : owned_alpha_g1_(std::move(owned_alpha_g1)),
        owned_beta_g2_(std::move(owned_beta_g2)),
        owned_gamma_g2_(std::move(owned_gamma_g2)),
        owned_delta_g2_(std::move(owned_delta_g2)),
        owned_l_g1_query_(std::move(owned_l_g1_query)) {
    SetParentValues();
  }
  OwnedVerifyingKey(const OwnedVerifyingKey& other)
      : owned_alpha_g1_(other.owned_alpha_g1_),
        owned_beta_g2_(other.owned_beta_g2_),
        owned_gamma_g2_(other.owned_gamma_g2_),
        owned_delta_g2_(other.owned_delta_g2_),
        owned_l_g1_query_(other.owned_l_g1_query_) {
    SetParentValues();
  }
  OwnedVerifyingKey& operator=(const OwnedVerifyingKey& other) {
    owned_alpha_g1_ = other.owned_alpha_g1_;
    owned_beta_g2_ = other.owned_beta_g2_;
    owned_gamma_g2_ = other.owned_gamma_g2_;
    owned_delta_g2_ = other.owned_delta_g2_;
    owned_l_g1_query_ = other.owned_l_g1_query_;
    SetParentValues();
    return *this;
  }
  OwnedVerifyingKey(OwnedVerifyingKey&& other)
      : owned_alpha_g1_(std::move(other.owned_alpha_g1_)),
        owned_beta_g2_(std::move(other.owned_beta_g2_)),
        owned_gamma_g2_(std::move(other.owned_gamma_g2_)),
        owned_delta_g2_(std::move(other.owned_delta_g2_)),
        owned_l_g1_query_(std::move(other.owned_l_g1_query_)) {
    SetParentValues();
  }
  OwnedVerifyingKey& operator=(OwnedVerifyingKey&& other) {
    owned_alpha_g1_ = std::move(other.owned_alpha_g1_);
    owned_beta_g2_ = std::move(other.owned_beta_g2_);
    owned_gamma_g2_ = std::move(other.owned_gamma_g2_);
    owned_delta_g2_ = std::move(other.owned_delta_g2_);
    owned_l_g1_query_ = std::move(other.owned_l_g1_query_);
    SetParentValues();
    return *this;
  }

  template <size_t MaxDegree, typename QAP>
  [[nodiscard]] bool Load(ToxicWaste<Curve>& toxic_waste,
                          const Circuit<F>& circuit) {
    KeyPreLoadResult<G1Point, MaxDegree> result;
    this->template PreLoad<QAP>(toxic_waste, circuit, &result);
    return Load(toxic_waste, result);
  }

  template <size_t MaxDegree>
  [[nodiscard]] bool Load(const ToxicWaste<Curve>& toxic_waste,
                          KeyPreLoadResult<G1Point, MaxDegree>& result) {
    F gamma_inverse = toxic_waste.gamma().Inverse();
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
      l[i] = this->ComputeABC(a[i], b[i], c[i], toxic_waste, gamma_inverse);
    }

    std::vector<G1JacobianPoint> l_g1_query_jacobian(l.size());
    if (!g1_msm.Run(l, &l_g1_query_jacobian)) return false;
    l.clear();

    owned_alpha_g1_ =
        (toxic_waste.alpha() * toxic_waste.g1_generator()).ToAffine();

    G2JacobianPoint beta_gamma_delta_jacobian[] = {
        toxic_waste.beta() * toxic_waste.g2_generator(),
        toxic_waste.gamma() * toxic_waste.g2_generator(),
        toxic_waste.delta() * toxic_waste.g2_generator(),
    };
    G2Point beta_gamma_delta[3];
    if (!G2JacobianPoint::BatchNormalize(beta_gamma_delta_jacobian,
                                         &beta_gamma_delta))
      return false;
    owned_beta_g2_ = std::move(beta_gamma_delta[0]);
    owned_gamma_g2_ = std::move(beta_gamma_delta[1]);
    owned_delta_g2_ = std::move(beta_gamma_delta[2]);

    owned_l_g1_query_.resize(l_g1_query_jacobian.size());
    if (!G1JacobianPoint::BatchNormalize(l_g1_query_jacobian,
                                         &owned_l_g1_query_))
      return false;
    SetParentValues();
    return true;
  }

  OwnedPreparedVerifyingKey<Curve> ToOwnedPreparedVerifyingKey() &&;

 private:
  void SetParentValues() {
    this->alpha_g1_ = &owned_alpha_g1_;
    this->beta_g2_ = &owned_beta_g2_;
    this->gamma_g2_ = &owned_gamma_g2_;
    this->delta_g2_ = &owned_delta_g2_;
    this->l_g1_query_ = owned_l_g1_query_;
  }

  // [α]₁
  G1Point owned_alpha_g1_;
  // [β]₂
  G2Point owned_beta_g2_;
  // [γ]₂
  G2Point owned_gamma_g2_;
  // [δ]₂
  G2Point owned_delta_g2_;
  // |l_g1_query_[i]| = [(β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / γ]₁
  std::vector<G1Point> owned_l_g1_query_;
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_OWNED_VERIFYING_KEY_H_
