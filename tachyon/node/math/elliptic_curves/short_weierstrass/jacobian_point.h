#ifndef TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
#define TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/node/base/node_module.h"

namespace tachyon::node::math {

template <typename JacobianPoint,
          typename BaseField = typename JacobianPoint::BaseField,
          typename Curve = typename JacobianPoint::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddJacobianPoint(NodeModule& m, std::string_view name) {
  m.NewClass<JacobianPoint>(name)
      .template AddConstructor<>()
      .template AddConstructor<const BaseField&, const BaseField&,
                               const BaseField&>()
      .AddStaticMethod("zero", &JacobianPoint::Zero)
      .AddStaticMethod("generator", &JacobianPoint::Generator)
      .AddStaticMethod("random", &JacobianPoint::Random)
      .AddReadOnlyProperty("x", &JacobianPoint::x)
      .AddReadOnlyProperty("y", &JacobianPoint::y)
      .AddReadOnlyProperty("z", &JacobianPoint::z)
      .AddMethod("isZero", &JacobianPoint::IsZero)
      .AddMethod("isOnCurve", &JacobianPoint::IsOnCurve)
      .AddMethod("toString", &JacobianPoint::ToString)
      .AddMethod("toHexString", &JacobianPoint::ToHexString, false)
      .AddMethod("eq", &JacobianPoint::operator==)
      .AddMethod("ne", &JacobianPoint::operator!=)
      .AddMethod("add",
                 &JacobianPoint::template operator+ <const JacobianPoint&>)
      .AddMethod("addMixed",
                 &JacobianPoint::template operator+ <const AffinePointTy&>)
      .AddMethod("sub",
                 &JacobianPoint::template operator- <const JacobianPoint&>)
      .AddMethod("subMixed",
                 &JacobianPoint::template operator- <const AffinePointTy&>)
      .AddMethod("negative",
                 static_cast<JacobianPoint (JacobianPoint::*)() const>(
                     &JacobianPoint::operator-))
      .AddMethod("double", &JacobianPoint::Double);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
