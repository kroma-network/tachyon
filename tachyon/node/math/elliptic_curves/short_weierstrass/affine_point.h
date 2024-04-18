#ifndef TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include "tachyon/node/base/node_module.h"

namespace tachyon::node::math {

template <typename AffinePoint,
          typename BaseField = typename AffinePoint::BaseField>
void AddAffinePoint(NodeModule& m, std::string_view name) {
  m.NewClass<AffinePoint>(name)
      .template AddConstructor<>()
      .template AddConstructor<const BaseField&, const BaseField&, bool>(false)
      .AddStaticMethod("zero", &AffinePoint::Zero)
      .AddStaticMethod("generator", &AffinePoint::Generator)
      .AddStaticMethod("random", &AffinePoint::Random)
      .AddReadOnlyProperty("x", &AffinePoint::x)
      .AddReadOnlyProperty("y", &AffinePoint::y)
      .AddReadOnlyProperty("infinity", &AffinePoint::infinity)
      .AddMethod("isZero", &AffinePoint::IsZero)
      .AddMethod("isOnCurve", &AffinePoint::IsOnCurve)
      .AddMethod("toString", &AffinePoint::ToString)
      .AddMethod("toHexString", &AffinePoint::ToHexString, false)
      .AddMethod("eq", &AffinePoint::operator==)
      .AddMethod("ne", &AffinePoint::operator!=)
      .AddMethod("add", &AffinePoint::template operator+ <const AffinePoint&>)
      .AddMethod("sub", &AffinePoint::template operator- <const AffinePoint&>)
      .AddMethod("negate", static_cast<AffinePoint (AffinePoint::*)() const>(
                               &AffinePoint::operator-))
      .AddMethod("double", &AffinePoint::Double);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
