#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"

#include <tuple>

#include "gtest/gtest.h"

namespace tachyon::math::bn254 {

TEST(BN254CurveTest, Compile) {
  BN254Curve curve;
  std::ignore = curve;
}

}  // namespace tachyon::math::bn254
