#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_G1_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_G1_H_

#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"
#include "tachyon/math/elliptic_curves/msm/glv.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

template <typename Fq, typename Fr>
class G1CurveConfig {
 public:
  using BaseField = Fq;
  using ScalarField = Fr;

  // clang-format off
  // Parameters are from https://zips.z.cash/protocol/protocol.pdf#page=98
  // A: Mont(0)
  constexpr static BigInt<6> kA = BigInt<6>(0);
  // B: Mont(4)
  constexpr static BigInt<6> kB = BigInt<6>({
    UINT64_C(12260768510540316659),
    UINT64_C(6038201419376623626),
    UINT64_C(5156596810353639551),
    UINT64_C(12813724723179037911),
    UINT64_C(10288881524157229871),
    UINT64_C(708830206584151678),
  });
  // X:
  //  Dec: 3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507
  //  Hex: 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
  // Y:
  //  Dec: 1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569
  //  Hex: 0x8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1
  constexpr static Point2<BigInt<6>> kGenerator = Point2<BigInt<6>>(
    BigInt<6>({
      UINT64_C(6679831729115696150),
      UINT64_C(8653662730902241269),
      UINT64_C(1535610680227111361),
      UINT64_C(17342916647841752903),
      UINT64_C(17135755455211762752),
      UINT64_C(1297449291367578485),
    }),
    BigInt<6>({
      UINT64_C(13451288730302620273),
      UINT64_C(10097742279870053774),
      UINT64_C(15949884091978425806),
      UINT64_C(5885175747529691540),
      UINT64_C(1016841820992199104),
      UINT64_C(845620083434234474),
    })
  );
  // Dec: 793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350
  // Hex: 0x5f19672fdf76ce51ba69c6076a0f77eaddb3a93be6f89688de17d813620a00022e01fffffffefffe
  constexpr static BigInt<6> kEndomorphismCoefficient = BigInt<6>({
    UINT64_C(3526659474838938856),
    UINT64_C(17562030475567847978),
    UINT64_C(1632777218702014455),
    UINT64_C(14009062335050482331),
    UINT64_C(3906511377122991214),
    UINT64_C(368068849512964448),
  });
  // Dec: 52435875175126190479447740508185965837461563690374988244538805122978187051009
  // Hex: 0x73eda753299d7d483339d80809a1d804a7780001fffcb7fcfffffffe00000001
  constexpr static BigInt<4> kLambda = BigInt<4>({
    UINT64_C(7865245318337523249),
    UINT64_C(18346590209729131401),
    UINT64_C(15545362854776399464),
    UINT64_C(6505881510324251116),
  });
  // Optimal decomposition as per Ch. 6.3.2: Decompositions for the k = 12 BLS
  // Family, from Guide to Pairing Based Cryptography by El Mrabet
  // Mont(X²)
  // Dec: 228988810152649578064853576960394133504
  // Hex: 0xac45a4010001a4020000000100000000
  constexpr static BigInt<4> kGLVCoeff00 = BigInt<4>({
    UINT64_C(10581498751077061072),
    UINT64_C(6134313272518502517),
    UINT64_C(6592600117572923804),
    UINT64_C(1847635349140198235),
  });
  // Mont(1)
  static constexpr BigInt<4> kGLVCoeff01 = ScalarField::Config::kOne;
  // Mont(-1)
  static constexpr BigInt<4> kGLVCoeff10 = BigInt<4>({
    UINT64_C(18446744060824649731),
    UINT64_C(18102478225614246908),
    UINT64_C(11073656695919314959),
    UINT64_C(6613806504683796440),
  });
  // Mont(X² - 1)
  // Dec: 228988810152649578064853576960394133503
  // Hex: 0xac45a4010001a40200000000ffffffff
  constexpr static BigInt<4> kGLVCoeff11 = BigInt<4>({
    UINT64_C(10581498742487126482),
    UINT64_C(18202632089594667123),
    UINT64_C(13975037914852467110),
    UINT64_C(107924994359545323),
  });
  // clang-format on
};

// TODO(chokobole): Enable this
// using G1AffinePoint = AffinePoint<SWCurve<G1CurveConfig<Fq, Fr>>>;
// using G1JacobianPoint = JacobianPoint<SWCurve<G1CurveConfig<Fq, Fr>>>;
#if defined(TACHYON_GMP_BACKEND)
using G1AffinePointGmp = AffinePoint<SWCurve<G1CurveConfig<FqGmp, FrGmp>>>;
using G1JacobianPointGmp = JacobianPoint<SWCurve<G1CurveConfig<FqGmp, FrGmp>>>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_G1_H_
