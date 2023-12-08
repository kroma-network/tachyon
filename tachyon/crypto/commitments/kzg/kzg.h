// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"

namespace tachyon {
namespace crypto {

template <typename G1PointTy, size_t MaxDegree,
          typename Commitment = typename math::Pippenger<G1PointTy>::Bucket>
class KZG {
 public:
  using Field = typename G1PointTy::ScalarField;

  static constexpr size_t kMaxDegree = MaxDegree;

  KZG() = default;

  KZG(std::vector<G1PointTy>&& g1_powers_of_tau,
      std::vector<G1PointTy>&& g1_powers_of_tau_lagrange)
      : g1_powers_of_tau_(std::move(g1_powers_of_tau)),
        g1_powers_of_tau_lagrange_(std::move(g1_powers_of_tau_lagrange)) {
    CHECK_EQ(g1_powers_of_tau_.size(), g1_powers_of_tau_lagrange_.size());
    CHECK_LE(g1_powers_of_tau_.size(), kMaxDegree + 1);
  }

  const std::vector<G1PointTy>& g1_powers_of_tau() const {
    return g1_powers_of_tau_;
  }

  const std::vector<G1PointTy>& g1_powers_of_tau_lagrange() const {
    return g1_powers_of_tau_lagrange_;
  }

  size_t N() const { return g1_powers_of_tau_.size(); }

  [[nodiscard]] bool UnsafeSetup(size_t size) {
    return UnsafeSetup(size, Field::Random());
  }

  [[nodiscard]] bool UnsafeSetup(size_t size, const Field& tau) {
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
    std::unique_ptr<DomainTy> domain = DomainTy::Create(size);
    typename DomainTy::DenseCoeffs lagrange_coeffs =
        domain->EvaluateAllLagrangeCoefficients(tau);
    std::vector<G1JacobianPointTy> g1_powers_of_tau_lagrange_jacobian;

    g1_powers_of_tau_lagrange_jacobian.resize(size);
    if (!G1PointTy::MultiScalarMul(lagrange_coeffs.coefficients(), g1,
                                   &g1_powers_of_tau_lagrange_jacobian)) {
      return false;
    }

    g1_powers_of_tau_lagrange_.resize(size);
    return math::ConvertPoints(g1_powers_of_tau_lagrange_jacobian,
                               &g1_powers_of_tau_lagrange_);
  }

  // Return false if |n| >= |N()|.
  [[nodiscard]] bool Downsize(size_t n) {
    if (n >= N()) return false;
    g1_powers_of_tau_.resize(n);
    g1_powers_of_tau_lagrange_.resize(n);
    return true;
  }

  template <typename BaseContainerTy>
  [[nodiscard]] bool Commit(const BaseContainerTy& v, Commitment* out) const {
    return DoMSM(g1_powers_of_tau_, v, out);
  }

  template <typename BaseContainerTy>
  [[nodiscard]] bool CommitLagrange(const BaseContainerTy& v,
                                    Commitment* out) const {
    return DoMSM(g1_powers_of_tau_lagrange_, v, out);
  }

 private:
  template <typename BaseContainerTy, typename ScalarContainerTy>
  static bool DoMSM(const BaseContainerTy& bases,
                    const ScalarContainerTy& scalars, Commitment* out) {
    using Bucket = typename math::Pippenger<G1PointTy>::Bucket;

    math::VariableBaseMSM<G1PointTy> msm;
    absl::Span<const G1PointTy> bases_span = absl::Span<const G1PointTy>(
        bases.data(), std::min(bases.size(), scalars.size()));
    if constexpr (std::is_same_v<Commitment, Bucket>) {
      return msm.Run(bases_span, scalars, out);
    } else {
      Bucket result;
      if (!msm.Run(bases_span, scalars, &result)) return false;
      *out = math::ConvertPoint<Commitment>(result);
      return true;
    }
  }

  std::vector<G1PointTy> g1_powers_of_tau_;
  std::vector<G1PointTy> g1_powers_of_tau_lagrange_;
};

}  // namespace crypto

namespace base {

template <typename G1PointTy, size_t MaxDegree, typename Commitment>
class Copyable<crypto::KZG<G1PointTy, MaxDegree, Commitment>> {
 public:
  using PCS = crypto::KZG<G1PointTy, MaxDegree, Commitment>;

  static bool WriteTo(const PCS& pcs, Buffer* buffer) {
    return buffer->WriteMany(pcs.g1_powers_of_tau(),
                             pcs.g1_powers_of_tau_lagrange());
  }

  static bool ReadFrom(const Buffer& buffer, PCS* pcs) {
    std::vector<G1PointTy> g1_powers_of_tau;
    std::vector<G1PointTy> g1_powers_of_tau_lagrange;
    if (!buffer.ReadMany(&g1_powers_of_tau, &g1_powers_of_tau_lagrange)) {
      return false;
    }

    *pcs =
        PCS(std::move(g1_powers_of_tau), std::move(g1_powers_of_tau_lagrange));
    return true;
  }

  static size_t EstimateSize(const PCS& pcs) {
    return base::EstimateSize(pcs.g1_powers_of_tau()) +
           base::EstimateSize(pcs.g1_powers_of_tau_lagrange());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_
