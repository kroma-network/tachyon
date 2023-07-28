#include "tachyon/math/elliptic_curves/secp/secp256k1/curve_config.h"

#include "gtest/gtest.h"

namespace tachyon::math {
namespace secp256k1 {

TEST(CurveConfig, Params) {
  auto a = CurveConfig<Fq, Fr>::kA;
  EXPECT_EQ(a, Fq(0).ToMontgomery());
  auto b = CurveConfig<Fq, Fr>::kB;
  EXPECT_EQ(b, Fq(7).ToMontgomery());
  auto generator = CurveConfig<Fq, Fr>::kGenerator;
  // clang-format off
  EXPECT_EQ(generator,
            Point2<BigInt<4>>(
                Fq::FromDecString("55066263022277343669578718895168534326250603453777594175500187360389116729240")
                    .ToMontgomery(),
                Fq::FromDecString("32670510020758816978083085130507043184471273380659243275938904335757337482424")
                    .ToMontgomery()));
  // clang-format on
}

}  // namespace secp256k1
}  // namespace tachyon::math
