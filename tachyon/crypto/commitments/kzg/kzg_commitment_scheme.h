// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_COMMITMENT_SCHEME_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/crypto/commitments/univariate_polynomial_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon {
namespace crypto {

template <typename G1PointTy, typename G2PointTy,
          typename ResultTy = typename math::Pippenger<G1PointTy>::Bucket>
class KZGCommitmentScheme
    : public UnivariatePolynomialCommitmentScheme<
          KZGCommitmentScheme<G1PointTy, G2PointTy, ResultTy>> {
 public:
  using Base = UnivariatePolynomialCommitmentScheme<
      KZGCommitmentScheme<G1PointTy, G2PointTy, ResultTy>>;
  using Field = typename G1PointTy::ScalarField;

  static constexpr size_t kMaxDegree = Base::kMaxSize - 1;

  KZGCommitmentScheme() = default;

  KZGCommitmentScheme(std::vector<G1PointTy> g1_powers_of_tau,
                      std::vector<G1PointTy> g1_powers_of_tau_lagrange,
                      G2PointTy tau_g2)
      : g1_powers_of_tau_(std::move(g1_powers_of_tau)),
        g1_powers_of_tau_lagrange_(std::move(g1_powers_of_tau_lagrange)),
        tau_g2_(std::move(tau_g2)) {
    CHECK_EQ(g1_powers_of_tau_.size(), g1_powers_of_tau_lagrange_.size());
    CHECK_LE(g1_powers_of_tau_.size(), Base::kMaxSize);
  }

  const std::vector<G1PointTy>& g1_powers_of_tau() const {
    return g1_powers_of_tau_;
  }

  const std::vector<G1PointTy>& g1_powers_of_tau_lagrange() const {
    return g1_powers_of_tau_lagrange_;
  }

  const G2PointTy& tau_g2() const { return tau_g2_; }

  // VectorCommitmentScheme methods
  size_t N() const { return g1_powers_of_tau_.size(); }

  [[nodiscard]] bool UnsafeSetupWithTau(size_t size, Field tau) {
    using G1JacobianPointTy = typename G1PointTy::JacobianPointTy;
    using DomainTy = math::UnivariateEvaluationDomain<Field, kMaxDegree>;

    // |g1_powers_of_tau_| = [𝜏⁰g₁, 𝜏¹g₁, ... , 𝜏ⁿ⁻¹g₁]
    G1PointTy g1 = G1PointTy::Generator();
    std::vector<Field> powers_of_tau = Field::GetSuccessivePowers(size, tau);
    std::vector<G1JacobianPointTy> g1_powers_of_tau_jacobian;

    g1_powers_of_tau_jacobian.resize(size);
    if (!G1PointTy::MultiScalarMul(powers_of_tau, g1,
                                   &g1_powers_of_tau_jacobian)) {
      return false;
    }
    g1_powers_of_tau_.resize(size);
    if (!math::ConvertPoints(g1_powers_of_tau_jacobian, &g1_powers_of_tau_)) {
      return false;
    }

    // Get |g1_powers_of_tau_lagrange_| from 𝜏 and g₁.
    std::unique_ptr<DomainTy> domain =
        math::UnivariateEvaluationDomainFactory<Field, kMaxDegree>::Create(
            size);
    typename DomainTy::DenseCoeffs lagrange_coeffs =
        domain->EvaluateAllLagrangeCoefficients(tau);
    std::vector<G1JacobianPointTy> g1_powers_of_tau_lagrange_jacobian;

    g1_powers_of_tau_lagrange_jacobian.resize(size);
    if (!G1PointTy::MultiScalarMul(lagrange_coeffs.coefficients(), g1,
                                   &g1_powers_of_tau_lagrange_jacobian)) {
      return false;
    }

    g1_powers_of_tau_lagrange_.resize(size);
    if (!math::ConvertPoints(g1_powers_of_tau_lagrange_jacobian,
                             &g1_powers_of_tau_lagrange_)) {
      return false;
    }

    // |tau_g2_| = 𝜏 * g₂
    tau_g2_ = (G2PointTy::Generator() * tau).ToAffine();

    return true;
  }

  // Return false if |n| >= |N()|.
  bool Downsize(size_t n) {
    if (n >= N()) return false;
    g1_powers_of_tau_.resize(n);
    g1_powers_of_tau_lagrange_.resize(n);
    return true;
  }

 private:
  friend class VectorCommitmentScheme<
      KZGCommitmentScheme<G1PointTy, G2PointTy, ResultTy>>;
  friend class UnivariatePolynomialCommitmentScheme<
      KZGCommitmentScheme<G1PointTy, G2PointTy, ResultTy>>;

  bool DoUnsafeSetup(size_t size) {
    return UnsafeSetupWithTau(size, Field::Random());
  }

  [[nodiscard]] bool DoCommit(const std::vector<Field>& v,
                              ResultTy* out) const {
    return DoMSM(g1_powers_of_tau_, v, out);
  }

  [[nodiscard]] bool DoCommitLagrange(const std::vector<Field>& v,
                                      ResultTy* out) const {
    return DoMSM(g1_powers_of_tau_lagrange_, v, out);
  }

  static bool DoMSM(const std::vector<G1PointTy>& bases,
                    const std::vector<Field>& scalars, ResultTy* out) {
    using Bucket = typename math::Pippenger<G1PointTy>::Bucket;

    math::VariableBaseMSM<G1PointTy> msm;
    if constexpr (std::is_same_v<ResultTy, Bucket>) {
      return msm.Run(bases, scalars, out);
    } else {
      Bucket result;
      return msm.Run(bases, scalars, &result);
      *out = math::ConvertPoint<ResultTy>(result);
    }
    return true;
  }

  std::vector<G1PointTy> g1_powers_of_tau_;
  std::vector<G1PointTy> g1_powers_of_tau_lagrange_;
  G2PointTy tau_g2_;
};

template <typename G1PointTy, typename G2PointTy, typename CommitmentTy>
struct VectorCommitmentSchemeTraits<
    KZGCommitmentScheme<G1PointTy, G2PointTy, CommitmentTy>> {
 public:
  using Field = typename G1PointTy::ScalarField;
  using ResultTy = CommitmentTy;

  constexpr static size_t kMaxSize = size_t{1} << Field::Config::kTwoAdicity;
  constexpr static bool kIsTransparent = false;
};

}  // namespace crypto

namespace base {

template <typename G1PointTy, typename G2PointTy, typename ResultTy>
class Copyable<crypto::KZGCommitmentScheme<G1PointTy, G2PointTy, ResultTy>> {
 public:
  using PCS = crypto::KZGCommitmentScheme<G1PointTy, G2PointTy, ResultTy>;

  static bool WriteTo(const PCS& pcs, Buffer* buffer) {
    return buffer->WriteMany(pcs.g1_powers_of_tau(),
                             pcs.g1_powers_of_tau_lagrange(), pcs.tau_g2());
  }

  static bool ReadFrom(const Buffer& buffer, PCS* pcs) {
    std::vector<G1PointTy> g1_powers_of_tau;
    std::vector<G1PointTy> g1_powers_of_tau_lagrange;
    G2PointTy tau_g2;
    if (!buffer.ReadMany(&g1_powers_of_tau, &g1_powers_of_tau_lagrange,
                         &tau_g2)) {
      return false;
    }

    *pcs = PCS(std::move(g1_powers_of_tau),
               std::move(g1_powers_of_tau_lagrange), std::move(tau_g2));
    return true;
  }

  static size_t EstimateSize(const PCS& pcs) {
    return base::EstimateSize(pcs.g1_powers_of_tau()) +
           base::EstimateSize(pcs.g1_powers_of_tau_lagrange()) +
           base::EstimateSize(pcs.tau_g2());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_COMMITMENT_SCHEME_H_
