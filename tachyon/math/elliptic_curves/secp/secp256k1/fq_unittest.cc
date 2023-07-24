#include "tachyon/math/elliptic_curves/secp/secp256k1/fq.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {
namespace secp256k1 {

TEST(Fq, Init) {
  Fq::Config::Init();
  EXPECT_EQ(Fq::Config::kOne, Fq::One().ToMontgomery());
}

}  // namespace secp256k1
}  // namespace math
}  // namespace tachyon
