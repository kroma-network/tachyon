// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_COMMITMENT_SCHEME_H_

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

namespace tachyon {
namespace crypto {

// A Pedersen commitment is a point on an elliptic curve that is
// cryptographically binding to data but hides it.
template <typename PointTy, size_t MaxSize,
          typename ResultTy = typename math::Pippenger<PointTy>::Bucket>
class PedersenCommitmentScheme
    : public VectorCommitmentScheme<
          PedersenCommitmentScheme<PointTy, MaxSize, ResultTy>> {
 public:
  using Field = typename PointTy::ScalarField;

  PedersenCommitmentScheme() = default;
  PedersenCommitmentScheme(const PointTy& h,
                           const std::vector<PointTy>& generators)
      : h_(h), generators_(generators) {
    CHECK_LE(generators_.size(), MaxSize);
  }
  PedersenCommitmentScheme(PointTy&& h, std::vector<PointTy>&& generators)
      : h_(h), generators_(std::move(generators)) {
    CHECK_LE(generators_.size(), MaxSize);
  }

  const PointTy& h() const { return h_; }
  const std::vector<PointTy>& generators() const { return generators_; }

  // VectorCommitmentScheme methods
  size_t N() const { return generators_.size(); }

  std::string ToString() const {
    std::stringstream ss;
    ss << "h: " << h_ << ", generators: " << VectorToString(generators_);
    return ss.str();
  }

 private:
  friend class VectorCommitmentScheme<
      PedersenCommitmentScheme<PointTy, MaxSize, ResultTy>>;

  bool DoSetup(size_t size) {
    // NOTE(leegwangwoon): For security, |Random| is used instead of
    // |CreatePseudoRandomPoints|.
    // See
    // https://research.nccgroup.com/2023/03/22/breaking-pedersen-hashes-in-practice/

    h_ = PointTy::Random();
    generators_ = base::CreateVector(size, []() { return PointTy::Random(); });
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
                ResultTy* out) const {
    using Bucket = typename math::Pippenger<PointTy>::Bucket;

    math::VariableBaseMSM<PointTy> msm;
    Bucket result;
    if (!msm.Run(generators_, v, &result)) return false;
    if constexpr (std::is_same_v<ResultTy, Bucket>) {
      *out = r * h_ + result;
    } else {
      *out = r * h_ + math::ConvertPoint<ResultTy>(result);
    }
    return true;
  }

  PointTy h_;
  std::vector<PointTy> generators_;
};

template <typename PointTy, size_t MaxSize, typename CommitmentTy>
struct VectorCommitmentSchemeTraits<
    PedersenCommitmentScheme<PointTy, MaxSize, CommitmentTy>> {
 public:
  constexpr static size_t kMaxSize = MaxSize;
  constexpr static bool kIsTransparent = true;

  using Field = typename PointTy::ScalarField;
  using ResultTy = CommitmentTy;
};

}  // namespace crypto

namespace base {

template <typename PointTy, size_t MaxSize, typename ResultTy>
class Copyable<crypto::PedersenCommitmentScheme<PointTy, MaxSize, ResultTy>> {
 public:
  using PCS = crypto::PedersenCommitmentScheme<PointTy, MaxSize, ResultTy>;

  static bool WriteTo(const PCS& pcs, Buffer* buffer) {
    return buffer->WriteMany(pcs.h(), pcs.generators());
  }

  static bool ReadFrom(const Buffer& buffer, PCS* pcs) {
    PointTy h;
    std::vector<PointTy> generators;
    if (!buffer.ReadMany(&h, &generators)) {
      return false;
    }

    *pcs = PCS(std::move(h), std::move(generators));
    return true;
  }

  static size_t EstimateSize(const PCS& pcs) {
    return base::EstimateSize(pcs.h()) + base::EstimateSize(pcs.generators());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_COMMITMENT_SCHEME_H_
