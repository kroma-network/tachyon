// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_OWNED_PROVING_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_OWNED_PROVING_KEY_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "tachyon/zk/r1cs/groth16/owned_verifying_key.h"
#include "tachyon/zk/r1cs/groth16/proving_key.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename Curve>
class OwnedProvingKey : public ProvingKey<Curve> {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using F = typename G1Point::ScalarField;

  OwnedProvingKey() = default;
  OwnedProvingKey(const OwnedVerifyingKey<Curve>& owned_verifying_key,
                  const G1Point& owned_beta_g1, const G1Point& owned_delta_g1,
                  const std::vector<G1Point>& owned_a_g1_query,
                  const std::vector<G1Point>& owned_b_g1_query,
                  const std::vector<G2Point>& owned_b_g2_query,
                  const std::vector<G1Point>& owned_h_g1_query,
                  const std::vector<G1Point>& owned_l_g1_query)
      : owned_verifying_key_(owned_verifying_key),
        owned_beta_g1_(owned_beta_g1),
        owned_delta_g1_(owned_delta_g1),
        owned_a_g1_query_(owned_a_g1_query),
        owned_b_g1_query_(owned_b_g1_query),
        owned_b_g2_query_(owned_b_g2_query),
        owned_h_g1_query_(owned_h_g1_query),
        owned_l_g1_query_(owned_l_g1_query) {
    SetParentValues();
  }
  OwnedProvingKey(OwnedVerifyingKey<Curve>&& owned_verifying_key,
                  G1Point&& owned_beta_g1, G1Point&& owned_delta_g1,
                  std::vector<G1Point>&& owned_a_g1_query,
                  std::vector<G1Point>&& owned_b_g1_query,
                  std::vector<G2Point>&& owned_b_g2_query,
                  std::vector<G1Point>&& owned_h_g1_query,
                  std::vector<G1Point>&& owned_l_g1_query)
      : owned_verifying_key_(std::move(owned_verifying_key)),
        owned_beta_g1_(std::move(owned_beta_g1)),
        owned_delta_g1_(std::move(owned_delta_g1)),
        owned_a_g1_query_(std::move(owned_a_g1_query)),
        owned_b_g1_query_(std::move(owned_b_g1_query)),
        owned_b_g2_query_(std::move(owned_b_g2_query)),
        owned_h_g1_query_(std::move(owned_h_g1_query)),
        owned_l_g1_query_(std::move(owned_l_g1_query)) {
    SetParentValues();
  }
  OwnedProvingKey(const OwnedProvingKey& other)
      : owned_verifying_key_(other.owned_verifying_key_),
        owned_beta_g1_(other.owned_beta_g1_),
        owned_delta_g1_(other.owned_delta_g1_),
        owned_a_g1_query_(other.owned_a_g1_query_),
        owned_b_g1_query_(other.owned_b_g1_query_),
        owned_b_g2_query_(other.owned_b_g2_query_),
        owned_h_g1_query_(other.owned_h_g1_query_),
        owned_l_g1_query_(other.owned_l_g1_query_) {
    SetParentValues();
  }
  OwnedProvingKey& operator=(const OwnedProvingKey& other) {
    owned_verifying_key_ = other.owned_verifying_key_;
    owned_beta_g1_ = other.owned_beta_g1_;
    owned_delta_g1_ = other.owned_delta_g1_;
    owned_a_g1_query_ = other.owned_a_g1_query_;
    owned_b_g1_query_ = other.owned_b_g1_query_;
    owned_b_g2_query_ = other.owned_b_g2_query_;
    owned_h_g1_query_ = other.owned_h_g1_query_;
    owned_l_g1_query_ = other.owned_l_g1_query;
    SetParentValues();
    return *this;
  }
  OwnedProvingKey(OwnedProvingKey&& other)
      : owned_verifying_key_(std::move(other.owned_verifying_key_)),
        owned_beta_g1_(std::move(other.owned_beta_g1_)),
        owned_delta_g1_(std::move(other.owned_delta_g1_)),
        owned_a_g1_query_(std::move(other.owned_a_g1_query_)),
        owned_b_g1_query_(std::move(other.owned_b_g1_query_)),
        owned_b_g2_query_(std::move(other.owned_b_g2_query_)),
        owned_h_g1_query_(std::move(other.owned_h_g1_query_)),
        owned_l_g1_query_(std::move(other.owned_l_g1_query_)) {
    SetParentValues();
  }
  OwnedProvingKey& operator=(OwnedProvingKey&& other) {
    owned_verifying_key_ = std::move(other.owned_verifying_key_);
    owned_beta_g1_ = std::move(other.owned_beta_g1_);
    owned_delta_g1_ = std::move(other.owned_delta_g1_);
    owned_a_g1_query_ = std::move(other.owned_a_g1_query_);
    owned_b_g1_query_ = std::move(other.owned_b_g1_query_);
    owned_b_g2_query_ = std::move(other.owned_b_g2_query_);
    owned_h_g1_query_ = std::move(other.owned_h_g1_query_);
    owned_l_g1_query_ = std::move(other.owned_l_g1_query_);
    SetParentValues();
    return *this;
  }

  const OwnedVerifyingKey<Curve>& owned_verifying_key() const {
    return owned_verifying_key_;
  }
  OwnedVerifyingKey<Curve>&& TakeOwnedVerifyingKey() && {
    return std::move(owned_verifying_key_);
  }

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
    this->template PreLoad<QAP>(toxic_waste, circuit, &result);

    QAPInstanceMapResult<F>& qap_instance_map_result =
        result.qap_instance_map_result;
    math::FixedBaseMSM<G1Point>& g1_msm = result.g1_msm;

    if (!owned_verifying_key_.Load(toxic_waste, result, gamma_delta_inverse[0]))
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
      l[i] = this->ComputeABC(
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

    owned_beta_g1_ = std::move(beta_delta[0]);
    owned_delta_g1_ = std::move(beta_delta[1]);

    owned_a_g1_query_.resize(a_g1_query_jacobian.size());
    if (!G1JacobianPoint::BatchNormalize(a_g1_query_jacobian,
                                         &owned_a_g1_query_))
      return false;
    owned_b_g1_query_.resize(b_g1_query_jacobian.size());
    if (!G1JacobianPoint::BatchNormalize(b_g1_query_jacobian,
                                         &owned_b_g1_query_))
      return false;
    owned_b_g2_query_.resize(b_g2_query_jacobian.size());
    if (!G2JacobianPoint::BatchNormalize(b_g2_query_jacobian,
                                         &owned_b_g2_query_))
      return false;
    owned_h_g1_query_.resize(h_g1_query_jacobian.size());
    if (!G1JacobianPoint::BatchNormalize(h_g1_query_jacobian,
                                         &owned_h_g1_query_))
      return false;
    owned_l_g1_query_.resize(l_g1_query_jacobian.size());
    if (!G1JacobianPoint::BatchNormalize(l_g1_query_jacobian,
                                         &owned_l_g1_query_))
      return false;
    SetParentValues();
    return true;
  }

 private:
  void SetParentValues() {
    this->verifying_key_ = owned_verifying_key_;
    this->beta_g1_ = &owned_beta_g1_;
    this->delta_g1_ = &owned_delta_g1_;
    this->a_g1_query_ = owned_a_g1_query_;
    this->b_g1_query_ = owned_b_g1_query_;
    this->b_g2_query_ = owned_b_g2_query_;
    this->h_g1_query_ = owned_h_g1_query_;
    this->l_g1_query_ = owned_l_g1_query_;
  }

  OwnedVerifyingKey<Curve> owned_verifying_key_;
  // [β]₁
  G1Point owned_beta_g1_;
  // [δ]₁
  G1Point owned_delta_g1_;
  // |owned_a_g1_query_[i]| = [aᵢ(x)]₁
  std::vector<G1Point> owned_a_g1_query_;
  // |b_g1_query_[i]| = [bᵢ(x)]₁
  std::vector<G1Point> owned_b_g1_query_;
  // |owned_b_g2_query_[i]| = [bᵢ(x)]₂
  std::vector<G2Point> owned_b_g2_query_;
  // |owned_h_g1_query_[i]| = [(xⁱ * t(x)) / δ]₁
  std::vector<G1Point> owned_h_g1_query_;
  // |owned_l_g1_query_[i]| = [(β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / δ]₁
  std::vector<G1Point> owned_l_g1_query_;
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_OWNED_PROVING_KEY_H_
