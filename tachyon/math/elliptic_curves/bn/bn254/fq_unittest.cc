#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {
namespace bn254 {

TEST(Fq, Init) {
  Fq::Config::Init();
  EXPECT_EQ(Fq::Config::kOne, Fq::One().ToMontgomery());
}

}  // namespace bn254
}  // namespace math
}  // namespace tachyon
