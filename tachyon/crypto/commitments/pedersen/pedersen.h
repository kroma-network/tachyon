#ifndef TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_
#define TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::crypto {

// A Pedersen commitment is a point on an elliptic curve that is
// cryptographically binding to data but hides it.
template <typename PointTy>
class PedersenParams {
 public:
  using Bucket = typename math::Pippenger<PointTy>::Bucket;
  using ScalarField = typename PointTy::ScalarField;

  PedersenParams() = default;
  PedersenParams(const PointTy& h, const std::vector<PointTy>& generators)
      : h_(h), generators_(generators) {}
  PedersenParams(PointTy&& h, std::vector<PointTy>&& generators)
      : h_(h), generators_(std::move(generators)) {}

  // Setup: |generators|, |h|
  // |generators| and |h| are randomly generated base points.
  const PointTy& h() const { return h_; }

  void set_h(const PointTy& h) { h_ = h; }

  const std::vector<PointTy>& generators() const { return generators_; }

  void set_generators(const std::vector<PointTy>& generators) {
    generators_ = generators;
  }

  void set_generators(std::vector<PointTy>&& generators) {
    generators_ = std::move(generators);
  }

  static PedersenParams Random(size_t max_size) {
    // NOTE(leegwangwoon): For security, |Random| is used instead of
    // |CreatePseudoRandomPoints|.
    // See
    // https://research.nccgroup.com/2023/03/22/breaking-pedersen-hashes-in-practice/

    std::vector<PointTy> generators =
        base::CreateVector(max_size, []() { return PointTy::Random(); });

    return {PointTy::Random(), std::move(generators)};
  }

  // Pedersen Commitment:
  // |h|⋅|r| + <|g|, |v|> = |h|⋅|r| + |g₀|⋅|v₀| + |g₁|⋅|v₁| + ... + |gₙ|⋅|vₙ|
  // - |h| is a randomly generated base point from Setup.
  // - |r| is a random value called the blinding factor.
  // - |g| denotes random |generators| in Setup params.
  // - |v| is a vector of values to be committed.
  template <typename R>
  bool Commit(const std::vector<ScalarField>& v, const ScalarField& r,
              R* out) const {
    math::VariableBaseMSM<PointTy> msm;
    Bucket generator_msm_v;
    if (!msm.Run(generators_, v, &generator_msm_v)) return false;

    *out = r * h_ + math::ConvertPoint<R>(generator_msm_v);
    return true;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "h: " << h_ << ", generators: [";
    for (size_t i = 0; i < generators_.size(); ++i) {
      ss << generators_[i];
      if (i != generators_.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    return ss.str();
  }

 private:
  PointTy h_;
  std::vector<PointTy> generators_;
};

};  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_
