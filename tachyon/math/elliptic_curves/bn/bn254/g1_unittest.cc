#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {
namespace bn254 {

TEST(G1CurveConfig, Params) {
  auto a = G1CurveConfig<Fq, Fr>::kA;
  EXPECT_EQ(a, Fq(0).ToMontgomery());
  auto b = G1CurveConfig<Fq, Fr>::kB;
  EXPECT_EQ(b, Fq(3).ToMontgomery());
  auto generator = G1CurveConfig<Fq, Fr>::kGenerator;
  EXPECT_EQ(generator,
            Point2<BigInt<4>>(Fq(1).ToMontgomery(), Fq(2).ToMontgomery()));
}

}  // namespace bn254
}  // namespace math
}  // namespace tachyon
