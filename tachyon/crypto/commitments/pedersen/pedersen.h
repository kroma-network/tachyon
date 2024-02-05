// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_
#define TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_

#include <stddef.h>

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
template <typename Point, size_t MaxSize,
          typename Commitment = typename math::Pippenger<Point>::Bucket>
class Pedersen final
    : public VectorCommitmentScheme<Pedersen<Point, MaxSize, Commitment>> {
 public:
  using Field = typename Point::ScalarField;
  using Bucket = typename math::Pippenger<Point>::Bucket;

  Pedersen() = default;
  Pedersen(const Point& h, const std::vector<Point>& generators)
      : h_(h), generators_(generators) {
    CHECK_LE(generators_.size(), MaxSize);
  }
  Pedersen(Point&& h, std::vector<Point>&& generators)
      : h_(h), generators_(std::move(generators)) {
    CHECK_LE(generators_.size(), MaxSize);
  }

  const Point& h() const { return h_; }
  const std::vector<Point>& generators() const { return generators_; }

  void ResizeBatchCommitments() {
    batch_commitments_.resize(this->batch_commitment_state_.batch_count);
  }

  std::vector<Commitment> GetBatchCommitments() {
    std::vector<Commitment> batch_commitments;
    if constexpr (std::is_same_v<Commitment, Bucket>) {
      batch_commitments = std::move(batch_commitments_);
    } else {
      batch_commitments.resize(batch_commitments_.size());
      CHECK(Bucket::BatchNormalize(batch_commitments_, &batch_commitments));
      batch_commitments_.clear();
    }
    this->batch_commitment_state_.Reset();
    return batch_commitments;
  }

  // VectorCommitmentScheme methods
  const char* Name() const { return "Pedersen"; }

  size_t N() const { return generators_.size(); }

  std::string ToString() const {
    std::stringstream ss;
    ss << "h: " << h_ << ", generators: " << VectorToString(generators_);
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
    math::VariableBaseMSM<Point> msm;
    Bucket result;
    if (!msm.Run(generators_, v, &result)) return false;
    if constexpr (std::is_same_v<Commitment, Bucket>) {
      *out = r * h_ + result;
    } else {
      *out = r * h_ + math::ConvertPoint<Commitment>(result);
    }
    return true;
  }

  bool DoCommit(const std::vector<Field>& v, const Field& r,
                BatchCommitmentState& state, size_t index) {
    math::VariableBaseMSM<Point> msm;
    if (batch_commitments_.size() != state.batch_count)
      batch_commitments_.resize(state.batch_count);
    return msm.Run(generators_, v, &batch_commitments_[index]);
  }

  Point h_;
  std::vector<Point> generators_;
  std::vector<Bucket> batch_commitments_;
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
    return base::EstimateSize(pcs.h()) + base::EstimateSize(pcs.generators());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_
