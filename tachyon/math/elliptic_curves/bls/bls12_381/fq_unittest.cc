#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

TEST(Fq, Init) {
  Fq::Config::Init();
  EXPECT_EQ(Fq::Config::kOne, Fq(1).ToMontgomery());
}

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon
