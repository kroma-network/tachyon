#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G1_TEST_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G1_TEST_H_

#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::c::math::bn254 {

class G1Test : public testing::Test {
 public:
  static void SetUpTestSuite() { tachyon_bn254_g1_init(); }
};

}  // namespace tachyon::c::math::bn254

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G1_TEST_H_
