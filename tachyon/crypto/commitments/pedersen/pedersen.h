// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_
#define TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_

#include <stddef.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/crypto/commitments/vector_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

#if TACHYON_CUDA
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_gpu.h"
#endif

namespace tachyon {
namespace crypto {

// A Pedersen commitment is a point on an elliptic curve that is
// cryptographically binding to data but hides it.
template <typename Point, size_t MaxSize,
          typename Commitment = typename math::Pippenger<Point>::Bucket>
class Pedersen final
    : public VectorCommitmentScheme<Pedersen<Point, MaxSize, Commitment>> {
 public:
  using Field = typename Point::ScalarField;
  using Bucket = typename math::Pippenger<Point>::Bucket;
  using Curve = typename Point::Curve;
  using AddResult =
      typename math::internal::AdditiveSemigroupTraits<Point>::ReturnTy;

  Pedersen() = default;
  Pedersen(const Point& h, const std::vector<Point>& generators)
      : h_(h), generators_(generators) {
    CHECK_LE(generators_.size(), MaxSize);
#if TACHYON_CUDA
    SetupForGpu();
#endif
  }
  Pedersen(Point&& h, std::vector<Point>&& generators)
      : h_(h), generators_(std::move(generators)) {
    CHECK_LE(generators_.size(), MaxSize);
#if TACHYON_CUDA
    SetupForGpu();
#endif
  }

  const Point& h() const { return h_; }
  const std::vector<Point>& generators() const { return generators_; }

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
        new math::VariableBaseMSMGpu<Point>(mem_pool_.get(), stream_.get()));
  }
#endif

  void ResizeBatchCommitments() {
    size_t size = this->batch_commitment_state_.batch_count;
#if TACHYON_CUDA
    if (msm_gpu_) {
      gpu_batch_commitments_.resize(size);
      return;
    }
#endif
    cpu_batch_commitments_.resize(size);
  }

  std::vector<Commitment> GetBatchCommitments() {
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
      } else {
        batch_commitments.resize(cpu_batch_commitments_.size());
        CHECK(
            Bucket::BatchNormalize(cpu_batch_commitments_, &batch_commitments));
        cpu_batch_commitments_.clear();
      }
#if TACHYON_CUDA
    }
#endif
    this->batch_commitment_state_.Reset();
    return batch_commitments;
  }

  // VectorCommitmentScheme methods
  const char* Name() const { return "Pedersen"; }

  size_t N() const { return generators_.size(); }

  std::string ToString() const {
    std::stringstream ss;
    ss << "h: " << h_ << ", generators: " << ContainerToString(generators_);
    return ss.str();
  }

 private:
  friend class VectorCommitmentScheme<Pedersen<Point, MaxSize, Commitment>>;

  bool DoSetup(size_t size) {
    // NOTE(leegwangwoon): For security, |Random| is used instead of
    // |CreatePseudoRandomPoints|.
    // See
    // https://research.nccgroup.com/2023/03/22/breaking-pedersen-hashes-in-practice/

    h_ = Point::Random();
    generators_ = base::CreateVector(size, []() { return Point::Random(); });
    return true;
  }

  // Pedersen Commitment:
  // clang-format off
  // |h|⋅|r| + <|g|, |v|> = |h|⋅|r| + |g₀|⋅|v₀| + |g₁|⋅|v₁| + ... + |gₙ₋₁|⋅|vₙ₋₁|
  // - |h| is a randomly generated base point from Setup.
  // - |r| is a random value called the blinding factor.
  // - |g| denotes random |generators| in Setup params.
  // - |v| is a vector of values to be committed.
  // clang-format on
  bool DoCommit(const std::vector<Field>& v, const Field& r,
                Commitment* out) const {
#if TACHYON_CUDA
    if (msm_gpu_) {
      math::ProjectivePoint<Curve> msm_result;
      absl::Span<const Point> bases_span = absl::Span<const Point>(
          generators_.data(), std::min(generators_.size(), v.size()));
      if (!msm_gpu_->Run(bases_span, v, &msm_result)) return false;
      *out = ComputeCommitment(r, msm_result);
      return true;
    }
#endif

    math::VariableBaseMSM<Point> msm;
    Bucket msm_result;
    if (!msm.Run(generators_, v, &msm_result)) return false;
    *out = ComputeCommitment(r, msm_result);
    return true;
  }

  bool DoCommit(const std::vector<Field>& v, const Field& r,
                BatchCommitmentState& state, size_t index) {
#if TACHYON_CUDA
    if (msm_gpu_) {
      absl::Span<const Point> bases_span = absl::Span<const Point>(
          generators_.data(), std::min(generators_.size(), v.size()));
      return msm_gpu_->Run(bases_span, v, &gpu_batch_commitments_[index]);
    }
#endif
    math::VariableBaseMSM<Point> msm;
    if (cpu_batch_commitments_.size() != state.batch_count)
      cpu_batch_commitments_.resize(state.batch_count);
    return msm.Run(generators_, v, &cpu_batch_commitments_[index]);
  }

  template <typename MSMResult>
  Commitment ComputeCommitment(const Field& r,
                               const MSMResult& msm_result) const {
    AddResult rh = r * h_;
    if constexpr (math::internal::SupportsAdd<MSMResult, AddResult>::value) {
      if constexpr (std::is_same_v<MSMResult, Commitment>) {
        return rh + msm_result;
      } else {
        return math::ConvertPoint<Commitment>(rh + msm_result);
      }
    } else {
      MSMResult result = math::ConvertPoint<MSMResult>(rh) + msm_result;
      if constexpr (std::is_same_v<MSMResult, Commitment>) {
        return result;
      } else {
        return math::ConvertPoint<Commitment>(result);
      }
    }
  }

  Point h_;
  std::vector<Point> generators_;
  std::vector<Bucket> cpu_batch_commitments_;
#if TACHYON_CUDA
  device::gpu::ScopedMemPool mem_pool_;
  device::gpu::ScopedStream stream_;
  std::unique_ptr<math::VariableBaseMSMGpu<Point>> msm_gpu_;
  std::vector<math::ProjectivePoint<Curve>> gpu_batch_commitments_;
#endif
};

template <typename Point, size_t MaxSize, typename _Commitment>
struct VectorCommitmentSchemeTraits<Pedersen<Point, MaxSize, _Commitment>> {
 public:
  constexpr static size_t kMaxSize = MaxSize;
  constexpr static bool kIsTransparent = true;
  constexpr static bool kSupportsBatchMode = true;

  using Field = typename Point::ScalarField;
  using Commitment = _Commitment;
};

}  // namespace crypto

namespace base {

template <typename Point, size_t MaxSize, typename Commitment>
class Copyable<crypto::Pedersen<Point, MaxSize, Commitment>> {
 public:
  using PCS = crypto::Pedersen<Point, MaxSize, Commitment>;

  static bool WriteTo(const PCS& pcs, Buffer* buffer) {
    return buffer->WriteMany(pcs.h(), pcs.generators());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, PCS* pcs) {
    Point h;
    std::vector<Point> generators;
    if (!buffer.ReadMany(&h, &generators)) {
      return false;
    }

    *pcs = PCS(std::move(h), std::move(generators));
    return true;
  }

  static size_t EstimateSize(const PCS& pcs) {
    return base::EstimateSize(pcs.h(), pcs.generators());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_
