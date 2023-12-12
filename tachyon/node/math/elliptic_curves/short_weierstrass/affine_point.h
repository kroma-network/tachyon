#ifndef TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include "tachyon/node/base/node_module.h"

namespace tachyon::node::math {

template <typename AffinePointTy,
          typename BaseField = typename AffinePointTy::BaseField>
void AddAffinePoint(NodeModule& m, std::string_view name) {
  m.NewClass<AffinePointTy>(name)
      .template AddConstructor<>()
      .template AddConstructor<const BaseField&, const BaseField&, bool>(false)
      .AddStaticMethod("zero", &AffinePointTy::Zero)
      .AddStaticMethod("generator", &AffinePointTy::Generator)
      .AddStaticMethod("random", &AffinePointTy::Random)
      .AddReadOnlyProperty("x", &AffinePointTy::x)
      .AddReadOnlyProperty("y", &AffinePointTy::y)
      .AddReadOnlyProperty("infinity", &AffinePointTy::infinity)
      .AddMethod("isZero", &AffinePointTy::IsZero)
      .AddMethod("isOnCurve", &AffinePointTy::IsOnCurve)
      .AddMethod("toString", &AffinePointTy::ToString)
      .AddMethod("toHexString", &AffinePointTy::ToHexString, false)
      .AddMethod("eq", &AffinePointTy::operator==)
      .AddMethod("ne", &AffinePointTy::operator!=)
      .AddMethod("add",
                 &AffinePointTy::template operator+ <const AffinePointTy&>)
      .AddMethod("sub",
                 &AffinePointTy::template operator- <const AffinePointTy&>)
      .AddMethod("negative",
                 static_cast<AffinePointTy (AffinePointTy::*)() const>(
                     &AffinePointTy::operator-))
      .AddMethod("double", &AffinePointTy::Double);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
