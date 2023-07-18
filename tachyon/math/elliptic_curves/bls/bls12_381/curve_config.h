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

    Fq g1_x = Fq::FromDecString(
        "3685416753713387016781088315183077757961620795782546409894578378688607"
        "59"
        "2378376318836054947676345821548104185464507");
    Fq g1_y = Fq::FromDecString(
        "1339506544944476473020471379941921221584933875938349620426543736416511"
        "42"
        "3956333506472724655353366534992391756441569");
    Config::Generator() = JacobianPoint<Config>::FromAffine(
        AffinePoint<Config>(std::move(g1_x), std::move(g1_y)));

    EndomorphismCoefficient() = Fq::FromDecString(
        "7934793907292155126213797016334214470608867402810604930104564874272816"
        "49"
        "075476305620758731620350");

    Lambda() = Fr::FromDecString(
        "5243587517512619047944774050818596583746156369037498824453880512297818"
        "70"
        "51009");

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
