#ifndef TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
#define TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/node/base/node_module.h"

namespace tachyon::node::math {

template <typename JacobianPointTy,
          typename BaseField = typename JacobianPointTy::BaseField,
          typename Curve = typename JacobianPointTy::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddJacobianPoint(NodeModule& m, std::string_view name) {
  m.NewClass<JacobianPointTy>(name)
      .template AddConstructor<>()
      .template AddConstructor<const BaseField&, const BaseField&,
                               const BaseField&>()
      .AddStaticMethod("zero", &JacobianPointTy::Zero)
      .AddStaticMethod("generator", &JacobianPointTy::Generator)
      .AddStaticMethod("random", &JacobianPointTy::Random)
      .AddReadOnlyProperty("x", &JacobianPointTy::x)
      .AddReadOnlyProperty("y", &JacobianPointTy::y)
      .AddReadOnlyProperty("z", &JacobianPointTy::z)
      .AddMethod("isZero", &JacobianPointTy::IsZero)
      .AddMethod("isOnCurve", &JacobianPointTy::IsOnCurve)
      .AddMethod("toString", &JacobianPointTy::ToString)
      .AddMethod("toHexString", &JacobianPointTy::ToHexString, false)
      .AddMethod("eq", &JacobianPointTy::operator==)
      .AddMethod("ne", &JacobianPointTy::operator!=)
      .AddMethod("add",
                 &JacobianPointTy::template operator+ <const JacobianPointTy&>)
      .AddMethod("addMixed",
                 &JacobianPointTy::template operator+ <const AffinePointTy&>)
      .AddMethod("sub",
                 &JacobianPointTy::template operator- <const JacobianPointTy&>)
      .AddMethod("subMixed",
                 &JacobianPointTy::template operator- <const AffinePointTy&>)
      .AddMethod("negative",
                 static_cast<JacobianPointTy (JacobianPointTy::*)() const>(
                     &JacobianPointTy::operator-))
      .AddMethod("double", &JacobianPointTy::Double);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
