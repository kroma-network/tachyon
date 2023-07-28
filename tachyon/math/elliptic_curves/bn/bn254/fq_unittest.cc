#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

#include "gtest/gtest.h"

namespace tachyon::math {
namespace bn254 {

TEST(Fq, Init) { EXPECT_EQ(Fq::Config::kOne, Fq(1).ToMontgomery()); }

}  // namespace bn254
}  // namespace tachyon::math
