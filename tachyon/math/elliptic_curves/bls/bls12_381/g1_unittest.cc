#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"

#include "gtest/gtest.h"

namespace tachyon::math {
namespace bls12_381 {

TEST(G1CurveConfig, Params) {
  auto a = G1CurveConfig<Fq, Fr>::kA;
  EXPECT_EQ(a, Fq(0).ToMontgomery());
  auto b = G1CurveConfig<Fq, Fr>::kB;
  EXPECT_EQ(b, Fq(4).ToMontgomery());
  auto generator = G1CurveConfig<Fq, Fr>::kGenerator;
  // clang-format off
  EXPECT_EQ(generator,
            Point2<BigInt<6>>(
                Fq::FromDecString(
                    "3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507")
                    .ToMontgomery(),
                Fq::FromDecString(
                    "1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569")
                    .ToMontgomery()));
  // clang-format on
}

TEST(G1CurveConfig, GLVParams) {
  // clang-format off
  auto endomorphism_coefficient = G1CurveConfig<Fq, Fr>::kEndomorphismCoefficient;
  EXPECT_EQ(endomorphism_coefficient, Fq::FromDecString("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350").ToMontgomery());
  auto lambda = G1CurveConfig<Fq, Fr>::kLambda;
  EXPECT_EQ(lambda, Fr::FromDecString("52435875175126190479447740508185965837461563690374988244538805122978187051009").ToMontgomery());
  auto c00 = G1CurveConfig<Fq, Fr>::kGLVCoeff00;
  EXPECT_EQ(c00, Fr::FromDecString("228988810152649578064853576960394133504").ToMontgomery());
  auto c01 = G1CurveConfig<Fq, Fr>::kGLVCoeff01;
  EXPECT_EQ(c01, Fr(1).ToMontgomery());
  auto c10 = G1CurveConfig<Fq, Fr>::kGLVCoeff10;
  EXPECT_EQ(c10, Fr(1).Negative().ToMontgomery());
  auto c11 = G1CurveConfig<Fq, Fr>::kGLVCoeff11;
  EXPECT_EQ(c11, Fr::FromDecString("228988810152649578064853576960394133503").ToMontgomery());
  // clang-format on
}

}  // namespace bls12_381
}  // namespace tachyon::math
