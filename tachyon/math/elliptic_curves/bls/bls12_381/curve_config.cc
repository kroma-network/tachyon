#include "tachyon/math/elliptic_curves/bls/bls12_381/curve_config.h"

#include "tachyon/base/no_destructor.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

// static
void CurveConfig::Init() {
  Fq::Config::Init();
  Fr::Config::Init();

  B() = Fq(4);

  Fq g1_x = Fq::FromDecString(
      "368541675371338701678108831518307775796162079578254640989457837868860759"
      "2378376318836054947676345821548104185464507");
  Fq g1_y = Fq::FromDecString(
      "133950654494447647302047137994192122158493387593834962042654373641651142"
      "3956333506472724655353366534992391756441569");
  Generator() = JacobianPoint<Config>::FromAffine(
      AffinePoint<Config>(std::move(g1_x), std::move(g1_y)));

  EndomorphismCoefficient() = Fq::FromDecString(
      "793479390729215512621379701633421447060886740281060493010456487427281649"
      "075476305620758731620350");

  Lambda() = Fr::FromDecString(
      "524358751751261904794477405081859658374615636903749882445388051229781870"
      "51009");

  // Optimal decomposition as per Ch. 6.3.2: Decompositions for the k = 12 BLS
  // Family, from Guide to Pairing Based Cryptography by El Mrabet
  ScalarDecompositionCoefficients() = GLV<CurveConfig>::Coefficients(
      // v_2 = (X², 1)
      gmp::FromDecString("228988810152649578064853576960394133504"),
      mpz_class(1),
      // v_1 = (-1, X² - 1)
      mpz_class(-1),
      gmp::FromDecString("228988810152649578064853576960394133503"));
}

// static
Fq& CurveConfig::EndomorphismCoefficient() {
  static base::NoDestructor<Fq> coeff;
  return *coeff;
}

// static
Fr& CurveConfig::Lambda() {
  static base::NoDestructor<Fr> lambda;
  return *lambda;
}

// static
GLV<CurveConfig>::Coefficients& CurveConfig::ScalarDecompositionCoefficients() {
  static base::NoDestructor<GLV<CurveConfig>::Coefficients> coefficients;
  return *coefficients;
}

// static
JacobianPoint<CurveConfig::Config> CurveConfig::Endomorphism(
    const JacobianPoint<Config>& point) {
  return JacobianPoint<Config>(point.x() * EndomorphismCoefficient(), point.y(),
                               point.z());
}

// static
AffinePoint<CurveConfig::Config> CurveConfig::EndomorphismAffine(
    const AffinePoint<Config>& point) {
  return AffinePoint<Config>(point.x() * EndomorphismCoefficient(), point.y());
}

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon
