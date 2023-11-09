#include "tachyon/math/elliptic_curves/bls12/bls12_381/bls12_381.h"

#include <tuple>

#include "gtest/gtest.h"

namespace tachyon::math::bls12_381 {

TEST(BLS12_381CurveTest, Compile) {
  BLS12_381Curve curve;
  std::ignore = curve;
}

}  // namespace tachyon::math::bls12_381
