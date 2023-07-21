#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_H_

#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"
#include "tachyon/math/elliptic_curves/msm/glv.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

template <typename Fq, typename Fr>
class CurveConfig : public SWCurveConfig<Fq, Fr> {
 public:
  using Config = SWCurveConfig<Fq, Fr>;

  static void Init() {
    Fq::Config::Init();
    Fr::Config::Init();

    Config::B() = Fq(4);

    // clang-format off
    // Parameters are from https://electriccoin.co/blog/new-snark-curve/
    
    // Dec: 3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507
    // Hex: 17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
    Fq g1_x = Fq::FromDecString("3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507");

    // Dec: 1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569
    // Hex: 8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1
    Fq g1_y = Fq::FromDecString("1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569");
    // clang-format on

    Config::Generator() = JacobianPoint<Config>::FromAffine(
        AffinePoint<Config>(std::move(g1_x), std::move(g1_y)));

    // clang-format off
    EndomorphismCoefficient() = Fq::FromDecString("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350");

    Lambda() = Fr::FromDecString("52435875175126190479447740508185965837461563690374988244538805122978187051009");
    // clang-format on

    // Optimal decomposition as per Ch. 6.3.2: Decompositions for the k = 12 BLS
    // Family, from Guide to Pairing Based Cryptography by El Mrabet
    ScalarDecompositionCoefficients() = typename GLV<CurveConfig>::Coefficients(
        // v_2 = (X², 1)
        gmp::FromDecString("228988810152649578064853576960394133504"),
        mpz_class(1),
        // v_1 = (-1, X² - 1)
        mpz_class(-1),
        gmp::FromDecString("228988810152649578064853576960394133503"));
  }

  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(Fq, EndomorphismCoefficient);
  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(Fr, Lambda);
  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(typename GLV<CurveConfig>::Coefficients,
                                        ScalarDecompositionCoefficients);

  static JacobianPoint<Config> Endomorphism(
      const JacobianPoint<Config>& point) {
    return JacobianPoint<Config>(point.x() * EndomorphismCoefficient(),
                                 point.y(), point.z());
  }
  static AffinePoint<Config> EndomorphismAffine(
      const AffinePoint<Config>& point) {
    return AffinePoint<Config>(point.x() * EndomorphismCoefficient(),
                               point.y());
  }
};

// TODO(chokobole): Enable this
// using G1AffinePoint = AffinePoint<CurveConfig<Fq, Fr>::Config>;
// using G1JacobianPoint = JacobianPoint<CurveConfig<Fq, Fr>::Config>;
#if defined(TACHYON_GMP_BACKEND)
using G1AffinePointGmp = AffinePoint<CurveConfig<FqGmp, FrGmp>::Config>;
using G1JacobianPointGmp = JacobianPoint<CurveConfig<FqGmp, FrGmp>::Config>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_H_
