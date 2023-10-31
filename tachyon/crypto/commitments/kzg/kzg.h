// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon {
namespace crypto {

template <typename G1PointTy, typename G2PointTy>
class KZGParams {
 public:
  using Field = typename G1PointTy::ScalarField;

  static constexpr size_t MaxDegree = size_t{1} << Field::Config::kTwoAdicity;

  KZGParams() = default;

  KZGParams(size_t k, std::vector<G1PointTy> g1_powers_of_tau,
            std::vector<G1PointTy> g1_powers_of_tau_lagrange, G2PointTy tau_g2)
      : k_(k),
        n_((base::CheckedNumeric<size_t>(1) << k).ValueOrDie()),
        g1_powers_of_tau_(std::move(g1_powers_of_tau)),
        g1_powers_of_tau_lagrange_(std::move(g1_powers_of_tau_lagrange)),
        tau_g2_(std::move(tau_g2)) {
    CHECK_LE(k, n_);
  }

  size_t k() const { return k_; }

  size_t n() const { return n_; }

  const std::vector<G1PointTy>& g1_powers_of_tau() const {
    return g1_powers_of_tau_;
  }

  const std::vector<G1PointTy>& g1_powers_of_tau_lagrange() const {
    return g1_powers_of_tau_lagrange_;
  }

  const G2PointTy& tau_g2() const { return tau_g2_; }

  // Initialize parameters with a random toxic point.
  // MUST NOT be used in production.
  [[nodiscard]] bool UnsafeSetup(size_t degree) {
    return UnsafeSetupWithTau(degree, Field::Random());
  }

  // Initialize parameters with given toxic point |tau|.
  // MUST NOT be used in production.
  [[nodiscard]] bool UnsafeSetupWithTau(size_t degree, Field tau) {
    // Get |n_| and |k_| from |degree|.
    k_ = degree;
    n_ = (base::CheckedNumeric<size_t>(1) << degree).ValueOrDie();

    // |g1_powers_of_tau_| = [𝜏⁰g₁, 𝜏¹g₁, ... , 𝜏ⁿ⁻¹g₁]
    G1PointTy g1 = G1PointTy::Generator();
    std::vector<Field> powers_of_tau = Field::GetSuccessivePowers(n_, tau);
    std::vector<typename G1PointTy::JacobianPointTy> g1_powers_of_tau_jacobian;

    g1_powers_of_tau_jacobian.resize(n_);
    if (!G1PointTy::MultiScalarMul(powers_of_tau, g1,
                                   &g1_powers_of_tau_jacobian)) {
      return false;
    }
    g1_powers_of_tau_.resize(n_);
    if (!math::ConvertPoints(g1_powers_of_tau_jacobian, &g1_powers_of_tau_)) {
      return false;
    }

    // Get |g1_powers_of_tau_lagrange_| from 𝜏 and g₁.
    std::unique_ptr<math::UnivariateEvaluationDomain<Field, MaxDegree>> domain =
        math::UnivariateEvaluationDomainFactory<Field, MaxDegree>::Create(n_);
    typename math::UnivariateEvaluationDomain<Field, MaxDegree>::DenseCoeffs
        lagrange_coeffs = domain->EvaluateAllLagrangeCoefficients(tau);
    std::vector<typename G1PointTy::JacobianPointTy>
        g1_powers_of_tau_lagrange_jacobian;

    g1_powers_of_tau_lagrange_jacobian.resize(n_);
    if (!G1PointTy::MultiScalarMul(lagrange_coeffs.coefficients(), g1,
                                   &g1_powers_of_tau_lagrange_jacobian))
      return false;

    g1_powers_of_tau_lagrange_.resize(n_);
    if (!math::ConvertPoints(g1_powers_of_tau_lagrange_jacobian,
                             &g1_powers_of_tau_lagrange_)) {
      return false;
    }

    // |tau_g2_| = 𝜏 * g₂
    tau_g2_ = g2_.ScalarMul(tau).ToAffine();

    return true;
  }

  void Downsize(size_t small_k) {
    DCHECK_LT(small_k, k_);
    k_ = small_k;
    n_ = (base::CheckedNumeric<size_t>(1) << k_).ValueOrDie();
    g1_powers_of_tau_.resize(n_);
    g1_powers_of_tau_lagrange_.resize(n_);
  }

  // Commit to a polynomial with given coefficients.
  [[nodiscard]] bool Commit(
      const math::UnivariateDensePolynomial<Field, MaxDegree>& poly,
      G1PointTy* out) const {
    return DoCommit(poly.coefficients().coefficients(), g1_powers_of_tau_, out);
  }

  // Commit to a polynomial with Lagrange coefficients.
  [[nodiscard]] bool CommitLagrange(
      const math::UnivariateEvaluations<Field, MaxDegree>& poly,
      G1PointTy* out) const {
    return DoCommit(poly.evaluations(), g1_powers_of_tau_lagrange_, out);
  }

 private:
  template <typename Poly>
  [[nodiscard]] bool DoCommit(const Poly& poly,
                              const std::vector<G1PointTy>& bases,
                              G1PointTy* out) const {
    math::VariableBaseMSM<G1PointTy> msm;
    typename math::Pippenger<G1PointTy>::Bucket bucket;
    if (!msm.Run(bases, poly, &bucket)) return false;

    *out = math::ConvertPoint<G1PointTy>(bucket);
    return true;
  }

  size_t k_ = 0;
  size_t n_ = 0;
  std::vector<G1PointTy> g1_powers_of_tau_;
  std::vector<G1PointTy> g1_powers_of_tau_lagrange_;
  G2PointTy g2_ = G2PointTy::Generator();
  G2PointTy tau_g2_ = G2PointTy::Generator();
};

}  // namespace crypto

namespace base {

template <typename G1PointTy, typename G2PointTy>
class Copyable<crypto::KZGParams<G1PointTy, G2PointTy>> {
 public:
  static bool WriteTo(const crypto::KZGParams<G1PointTy, G2PointTy>& params,
                      Buffer* buffer) {
    return buffer->WriteMany(params.k(), params.n(), params.g1_powers_of_tau(),
                             params.g1_powers_of_tau_lagrange(),
                             params.tau_g2());
  }

  static bool ReadFrom(const Buffer& buffer,
                       crypto::KZGParams<G1PointTy, G2PointTy>* params) {
    size_t k;
    size_t n;
    std::vector<G1PointTy> g1_powers_of_tau;
    std::vector<G1PointTy> g1_powers_of_tau_lagrange;
    G2PointTy tau_g2;

    if (!buffer.ReadMany(&k, &n, &g1_powers_of_tau, &g1_powers_of_tau_lagrange,
                         &tau_g2)) {
      return false;
    }

    *params = crypto::KZGParams<G1PointTy, G2PointTy>(
        k, std::move(g1_powers_of_tau), std::move(g1_powers_of_tau_lagrange),
        std::move(tau_g2));
    return true;
  }

  static size_t EstimateSize(
      const crypto::KZGParams<G1PointTy, G2PointTy>& params) {
    return base::EstimateSize(params.k()) + base::EstimateSize(params.n()) +
           base::EstimateSize(params.g1_powers_of_tau()) +
           base::EstimateSize(params.g1_powers_of_tau_lagrange()) +
           base::EstimateSize(params.tau_g2());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_
