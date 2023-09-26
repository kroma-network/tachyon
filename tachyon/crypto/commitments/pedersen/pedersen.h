#ifndef TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_
#define TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_

#include <sstream>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::math {

template <typename PointTy>
class PedersenParams {
 public:
  using Bucket = typename Pippenger<PointTy>::Bucket;
  using ScalarField = typename PointTy::ScalarField;

  PedersenParams() = default;
  PedersenParams(const PointTy& h, const std::vector<PointTy>& generators)
      : h_(h), generators_(generators) {}
  PedersenParams(PointTy&& h, std::vector<PointTy>&& generators)
      : h_(h), generators_(std::move(generators)) {}

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

  // h⋅r + <g, v>
  template <typename R>
  bool Commit(const std::vector<ScalarField>& v, const ScalarField& r,
              R* out) const {
    VariableBaseMSM<PointTy> msm;
    Bucket generator_msm_v;
    if (!msm.Run(generators_, v, &generator_msm_v)) return false;

    *out = r * h_ + ConvertPoint<R>(generator_msm_v);
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

template <typename PointTy>
std::ostream& operator<<(std::ostream& os, const PedersenParams<PointTy>& p) {
  return os << p.ToString();
}

};  // namespace tachyon::math

#endif  // TACHYON_CRYPTO_COMMITMENTS_PEDERSEN_PEDERSEN_H_
