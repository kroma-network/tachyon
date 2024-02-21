// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/crypto/commitments/batch_commitment_state.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"

namespace tachyon {
namespace crypto {

template <typename G1Point, size_t MaxDegree,
          typename Commitment = typename math::Pippenger<G1Point>::Bucket>
class KZG {
 public:
  using Field = typename G1Point::ScalarField;
  using Bucket = typename math::Pippenger<G1Point>::Bucket;

  static constexpr size_t kMaxDegree = MaxDegree;

  KZG() = default;

  KZG(std::vector<G1Point>&& g1_powers_of_tau,
      std::vector<G1Point>&& g1_powers_of_tau_lagrange)
      : g1_powers_of_tau_(std::move(g1_powers_of_tau)),
        g1_powers_of_tau_lagrange_(std::move(g1_powers_of_tau_lagrange)) {
    CHECK_EQ(g1_powers_of_tau_.size(), g1_powers_of_tau_lagrange_.size());
    CHECK_LE(g1_powers_of_tau_.size(), kMaxDegree + 1);
  }

  const std::vector<G1Point>& g1_powers_of_tau() const {
    return g1_powers_of_tau_;
  }

  const std::vector<G1Point>& g1_powers_of_tau_lagrange() const {
    return g1_powers_of_tau_lagrange_;
  }

  void ResizeBatchCommitments(size_t size) { batch_commitments_.resize(size); }

  std::vector<Commitment> GetBatchCommitments(BatchCommitmentState& state) {
    std::vector<Commitment> batch_commitments;
    if constexpr (std::is_same_v<Commitment, Bucket>) {
      batch_commitments = std::move(batch_commitments_);
    } else {
      batch_commitments.resize(batch_commitments_.size());
      CHECK(Bucket::BatchNormalize(batch_commitments_, &batch_commitments));
      batch_commitments_.clear();
    }
    state.Reset();
    return batch_commitments;
  }

  size_t N() const { return g1_powers_of_tau_.size(); }

  [[nodiscard]] bool UnsafeSetup(size_t size) {
    return UnsafeSetup(size, Field::Random());
  }

  [[nodiscard]] bool UnsafeSetup(size_t size, const Field& tau) {
    using Domain = math::UnivariateEvaluationDomain<Field, kMaxDegree>;

    // |g1_powers_of_tau_| = [𝜏⁰g₁, 𝜏¹g₁, ... , 𝜏ⁿ⁻¹g₁]
    G1Point g1 = G1Point::Generator();
    std::vector<Field> powers_of_tau = Field::GetSuccessivePowers(size, tau);

    g1_powers_of_tau_.resize(size);
    if (!G1Point::BatchMapScalarFieldToPoint(g1, powers_of_tau,
                                             &g1_powers_of_tau_)) {
      return false;
    }

    // Get |g1_powers_of_tau_lagrange_| from 𝜏 and g₁.
    std::unique_ptr<Domain> domain = Domain::Create(size);
    std::vector<Field> lagrange_coeffs =
        domain->EvaluateAllLagrangeCoefficients(tau);

    g1_powers_of_tau_lagrange_.resize(size);
    return G1Point::BatchMapScalarFieldToPoint(g1, lagrange_coeffs,
                                               &g1_powers_of_tau_lagrange_);
  }

  // Return false if |n| >= |N()|.
  [[nodiscard]] bool Downsize(size_t n) {
    if (n >= N()) return false;
    g1_powers_of_tau_.resize(n);
    g1_powers_of_tau_lagrange_.resize(n);
    return true;
  }

  template <typename ScalarContainer>
  [[nodiscard]] bool Commit(const ScalarContainer& v, Commitment* out) const {
    return DoMSM(g1_powers_of_tau_, v, out);
  }

  template <typename ScalarContainer>
  [[nodiscard]] bool Commit(const ScalarContainer& v,
                            BatchCommitmentState& state, size_t index) {
    return DoMSM(g1_powers_of_tau_, v, state, index);
  }

  template <typename ScalarContainer>
  [[nodiscard]] bool CommitLagrange(const ScalarContainer& v,
                                    Commitment* out) const {
    return DoMSM(g1_powers_of_tau_lagrange_, v, out);
  }

  template <typename ScalarContainer>
  [[nodiscard]] bool CommitLagrange(const ScalarContainer& v,
                                    BatchCommitmentState& state, size_t index) {
    return DoMSM(g1_powers_of_tau_lagrange_, v, state, index);
  }

 private:
  template <typename BaseContainer, typename ScalarContainer>
  static bool DoMSM(const BaseContainer& bases, const ScalarContainer& scalars,
                    Commitment* out) {
    math::VariableBaseMSM<G1Point> msm;
    absl::Span<const G1Point> bases_span = absl::Span<const G1Point>(
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

  template <typename BaseContainer, typename ScalarContainer>
  bool DoMSM(const BaseContainer& bases, const ScalarContainer& scalars,
             BatchCommitmentState& state, size_t index) {
    math::VariableBaseMSM<G1Point> msm;
    absl::Span<const G1Point> bases_span = absl::Span<const G1Point>(
        bases.data(), std::min(bases.size(), scalars.size()));
    return msm.Run(bases_span, scalars, &batch_commitments_[index]);
  }

  std::vector<G1Point> g1_powers_of_tau_;
  std::vector<G1Point> g1_powers_of_tau_lagrange_;
  std::vector<Bucket> batch_commitments_;
};

}  // namespace crypto

namespace base {

template <typename G1Point, size_t MaxDegree, typename Commitment>
class Copyable<crypto::KZG<G1Point, MaxDegree, Commitment>> {
 public:
  using PCS = crypto::KZG<G1Point, MaxDegree, Commitment>;

  static bool WriteTo(const PCS& pcs, Buffer* buffer) {
    return buffer->WriteMany(pcs.g1_powers_of_tau(),
                             pcs.g1_powers_of_tau_lagrange());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, PCS* pcs) {
    std::vector<G1Point> g1_powers_of_tau;
    std::vector<G1Point> g1_powers_of_tau_lagrange;
    if (!buffer.ReadMany(&g1_powers_of_tau, &g1_powers_of_tau_lagrange)) {
      return false;
    }

    *pcs =
        PCS(std::move(g1_powers_of_tau), std::move(g1_powers_of_tau_lagrange));
    return true;
  }

  static size_t EstimateSize(const PCS& pcs) {
    return base::EstimateSize(pcs.g1_powers_of_tau(),
                              pcs.g1_powers_of_tau_lagrange());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_
