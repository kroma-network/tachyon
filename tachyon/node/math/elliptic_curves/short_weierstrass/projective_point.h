#ifndef TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
#define TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/node/base/node_module.h"

namespace tachyon::node::math {

template <typename ProjectivePoint,
          typename BaseField = typename ProjectivePoint::BaseField,
          typename Curve = typename ProjectivePoint::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddProjectivePoint(NodeModule& m, std::string_view name) {
  m.NewClass<ProjectivePoint>(name)
      .template AddConstructor<>()
      .template AddConstructor<const BaseField&, const BaseField&,
                               const BaseField&>()
      .AddStaticMethod("zero", &ProjectivePoint::Zero)
      .AddStaticMethod("generator", &ProjectivePoint::Generator)
      .AddStaticMethod("random", &ProjectivePoint::Random)
      .AddReadOnlyProperty("x", &ProjectivePoint::x)
      .AddReadOnlyProperty("y", &ProjectivePoint::y)
      .AddReadOnlyProperty("z", &ProjectivePoint::z)
      .AddMethod("isZero", &ProjectivePoint::IsZero)
      .AddMethod("isOnCurve", &ProjectivePoint::IsOnCurve)
      .AddMethod("toString", &ProjectivePoint::ToString)
      .AddMethod("toHexString", &ProjectivePoint::ToHexString, false)
      .AddMethod("eq", &ProjectivePoint::operator==)
      .AddMethod("ne", &ProjectivePoint::operator!=)
      .AddMethod("add",
                 &ProjectivePoint::template operator+ <const ProjectivePoint&>)
      .AddMethod("addMixed",
                 &ProjectivePoint::template operator+ <const AffinePointTy&>)
      .AddMethod("sub",
                 &ProjectivePoint::template operator- <const ProjectivePoint&>)
      .AddMethod("subMixed",
                 &ProjectivePoint::template operator- <const AffinePointTy&>)
      .AddMethod("negative",
                 static_cast<ProjectivePoint (ProjectivePoint::*)() const>(
                     &ProjectivePoint::operator-))
      .AddMethod("double", &ProjectivePoint::Double);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
