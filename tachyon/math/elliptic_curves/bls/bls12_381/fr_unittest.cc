#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

TEST(Fr, Init) {
  Fr::Config::Init();
  EXPECT_EQ(Fr::Config::kOne, Fr::One().ToMontgomery());
}

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon
