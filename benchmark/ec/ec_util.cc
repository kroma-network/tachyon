#include "benchmark/ec/ec_util.h"

namespace tachyon {

std::vector<math::bn254::G1AffinePoint> CreateRandomBn254Points(uint64_t nums) {
  using namespace math;
  std::vector<bn254::G1AffinePoint> ret;
  ret.reserve(nums);
  bn254::G1JacobianPoint p = bn254::G1JacobianPoint::Random();
  for (uint64_t i = 0; i < nums; ++i) {
    ret.push_back(p.DoubleInPlace().ToAffine());
  }
  return ret;
}

std::vector<math::bn254::Fr> CreateRandomBn254Scalars(uint64_t nums) {
  using namespace math;
  std::vector<bn254::Fr> ret;
  ret.reserve(nums);
  for (uint64_t i = 0; i < nums; ++i) {
    ret.push_back(bn254::Fr::Random());
  }
  return ret;
}

}  // namespace tachyon
