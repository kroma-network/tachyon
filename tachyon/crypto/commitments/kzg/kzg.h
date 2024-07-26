// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_H_

#include <stddef.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/crypto/commitments/batch_commitment_state.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/geometry/point_conversions.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"

#if TACHYON_CUDA
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_gpu.h"
#endif

namespace tachyon {
namespace zk {

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename Commitment>
class GWCExtension;

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename Commitment>
class SHPlonkExtension;

}  // namespace zk

namespace crypto {

template <typename G1Point, size_t MaxDegree,
          typename Commitment = typename math::Pippenger<G1Point>::Bucket>
class KZG {
 public:
  using Field = typename G1Point::ScalarField;
  using Bucket = typename math::Pippenger<G1Point>::Bucket;
  using Curve = typename G1Point::Curve;

  static constexpr size_t kMaxDegree = MaxDegree;

  KZG() = default;

  KZG(std::pmr::vector<G1Point>&& g1_powers_of_tau,
      std::pmr::vector<G1Point>&& g1_powers_of_tau_lagrange)
      : g1_powers_of_tau_(std::move(g1_powers_of_tau)),
        g1_powers_of_tau_lagrange_(std::move(g1_powers_of_tau_lagrange)) {
    CHECK_EQ(g1_powers_of_tau_.size(), g1_powers_of_tau_lagrange_.size());
    CHECK_LE(g1_powers_of_tau_.size(), kMaxDegree + 1);
#if TACHYON_CUDA
    SetupForGpu();
#endif
  }

  const std::pmr::vector<G1Point>& g1_powers_of_tau() const {
    return g1_powers_of_tau_;
  }

  const std::pmr::vector<G1Point>& g1_powers_of_tau_lagrange() const {
    return g1_powers_of_tau_lagrange_;
  }

#if TACHYON_CUDA
  void SetupForGpu() {
    if (msm_gpu_) return;

    gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                             gpuMemHandleTypeNone,
                             {gpuMemLocationTypeDevice, 0}};
    mem_pool_ = device::gpu::CreateMemPool(&props);

    uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
    gpuError_t error = gpuMemPoolSetAttribute(
        mem_pool_.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
    CHECK_EQ(error, gpuSuccess);
    stream_ = device::gpu::CreateStream();

    msm_gpu_.reset(
        new math::VariableBaseMSMGpu<G1Point>(mem_pool_.get(), stream_.get()));
  }
#endif

  void ResizeBatchCommitments(size_t size) {
#if TACHYON_CUDA
    if (msm_gpu_) {
      gpu_batch_commitments_.resize(size);
      return;
    }
#endif
    cpu_batch_commitments_.resize(size);
  }

  std::vector<Commitment> GetBatchCommitments(BatchCommitmentState& state) {
    std::vector<Commitment> batch_commitments;
#if TACHYON_CUDA
    if (msm_gpu_) {
      if constexpr (std::is_same_v<Commitment, math::ProjectivePoint<Curve>>) {
        batch_commitments = std::move(gpu_batch_commitments_);
        // NOLINTNEXTLINE(readability/braces)
      } else if constexpr (std::is_same_v<Commitment,
                                          math::AffinePoint<Curve>>) {
        batch_commitments.resize(gpu_batch_commitments_.size());
        CHECK(math::ProjectivePoint<Curve>::BatchNormalize(
            gpu_batch_commitments_, &batch_commitments));
        gpu_batch_commitments_.clear();
      } else {
        batch_commitments.resize(gpu_batch_commitments_.size());
        CHECK(math::ConvertPoints(gpu_batch_commitments_, &batch_commitments));
        gpu_batch_commitments_.clear();
      }
    } else {
#endif
      if constexpr (std::is_same_v<Commitment, Bucket>) {
        batch_commitments = std::move(cpu_batch_commitments_);
        // NOLINTNEXTLINE(readability/braces)
      } else if constexpr (std::is_same_v<Commitment,
                                          math::AffinePoint<Curve>>) {
        batch_commitments.resize(cpu_batch_commitments_.size());
        CHECK(
            Bucket::BatchNormalize(cpu_batch_commitments_, &batch_commitments));
        cpu_batch_commitments_.clear();
      } else {
        batch_commitments.resize(cpu_batch_commitments_.size());
        CHECK(math::ConvertPoints(cpu_batch_commitments_, &batch_commitments));
        cpu_batch_commitments_.clear();
      }
#if TACHYON_CUDA
    }
#endif
    state.Reset();
    return batch_commitments;
  }

  size_t N() const { return g1_powers_of_tau_.size(); }

  [[nodiscard]] bool UnsafeSetup(size_t size) {
    return UnsafeSetup(size, Field::Random());
  }

  [[nodiscard]] bool UnsafeSetup(size_t size, const Field& tau) {
    using Domain = math::UnivariateEvaluationDomain<Field, kMaxDegree>;

    // |g1_powers_of_tau_| = [τ⁰g₁, τ¹g₁, ... , τⁿ⁻¹g₁]
    G1Point g1 = G1Point::Generator();
    std::pmr::vector<Field> powers_of_tau =
        Field::GetSuccessivePowers(size, tau);

    g1_powers_of_tau_.resize(size);
    if (!G1Point::BatchMapScalarFieldToPoint(g1, powers_of_tau,
                                             &g1_powers_of_tau_)) {
      return false;
    }

    // Get |g1_powers_of_tau_lagrange_| from τ and g₁.
    std::unique_ptr<Domain> domain = Domain::Create(size);
    std::pmr::vector<Field> lagrange_coeffs =
        domain->EvaluateAllLagrangeCoefficients(tau);

    g1_powers_of_tau_lagrange_.resize(size);
    if (!G1Point::BatchMapScalarFieldToPoint(g1, lagrange_coeffs,
                                             &g1_powers_of_tau_lagrange_)) {
      return false;
    }

#if TACHYON_CUDA
    SetupForGpu();
#endif
    return true;
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
  template <typename, size_t, size_t, typename>
  friend class tachyon::zk::GWCExtension;

  template <typename, size_t, size_t, typename>
  friend class tachyon::zk::SHPlonkExtension;

  template <typename BaseContainer, typename ScalarContainer,
            typename OutCommitment>
  bool DoMSM(const BaseContainer& bases, const ScalarContainer& scalars,
             OutCommitment* out) const {
#if TACHYON_CUDA
    if (msm_gpu_) {
      absl::Span<const G1Point> bases_span = absl::Span<const G1Point>(
          bases.data(), std::min(bases.size(), scalars.size()));
      if constexpr (std::is_same_v<OutCommitment,
                                   math::ProjectivePoint<Curve>>) {
        return msm_gpu_->Run(bases_span, scalars, out);
      } else {
        math::ProjectivePoint<Curve> result;
        if (!msm_gpu_->Run(bases_span, scalars, &result)) return false;
        *out = math::ConvertPoint<OutCommitment>(result);
        return true;
      }
    }
#endif
    math::VariableBaseMSM<G1Point> msm;
    absl::Span<const G1Point> bases_span = absl::Span<const G1Point>(
        bases.data(), std::min(bases.size(), scalars.size()));
    if constexpr (std::is_same_v<OutCommitment, Bucket>) {
      return msm.Run(bases_span, scalars, out);
    } else {
      Bucket result;
      if (!msm.Run(bases_span, scalars, &result)) return false;
      *out = math::ConvertPoint<OutCommitment>(result);
      return true;
    }
  }

  template <typename BaseContainer, typename ScalarContainer>
  bool DoMSM(const BaseContainer& bases, const ScalarContainer& scalars,
             BatchCommitmentState& state, size_t index) {
#if TACHYON_CUDA
    if (msm_gpu_) {
      absl::Span<const G1Point> bases_span = absl::Span<const G1Point>(
          bases.data(), std::min(bases.size(), scalars.size()));
      return msm_gpu_->Run(bases_span, scalars, &gpu_batch_commitments_[index]);
    }
#endif
    math::VariableBaseMSM<G1Point> msm;
    absl::Span<const G1Point> bases_span = absl::Span<const G1Point>(
        bases.data(), std::min(bases.size(), scalars.size()));
    return msm.Run(bases_span, scalars, &cpu_batch_commitments_[index]);
  }

  std::pmr::vector<G1Point> g1_powers_of_tau_;
  std::pmr::vector<G1Point> g1_powers_of_tau_lagrange_;
  std::vector<Bucket> cpu_batch_commitments_;
#if TACHYON_CUDA
  device::gpu::ScopedMemPool mem_pool_;
  device::gpu::ScopedStream stream_;
  std::unique_ptr<math::VariableBaseMSMGpu<G1Point>> msm_gpu_;
  std::vector<math::ProjectivePoint<Curve>> gpu_batch_commitments_;
#endif
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
    std::pmr::vector<G1Point> g1_powers_of_tau;
    std::pmr::vector<G1Point> g1_powers_of_tau_lagrange;
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
