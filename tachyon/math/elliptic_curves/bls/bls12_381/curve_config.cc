#include "tachyon/math/elliptic_curves/bls/bls12_381/curve_config.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

// static
void CurveConfig::Init() {
  B() = Fq(4);

  Fq g1_x = Fq::FromDecString(
      "368541675371338701678108831518307775796162079578254640989457837868860759"
      "2378376318836054947676345821548104185464507");
  Fq g1_y = Fq::FromDecString(
      "133950654494447647302047137994192122158493387593834962042654373641651142"
      "3956333506472724655353366534992391756441569");
  Generator() = JacobianPoint<Config>::FromAffine(
      AffinePoint<Config>(std::move(g1_x), std::move(g1_y)));
}

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon
