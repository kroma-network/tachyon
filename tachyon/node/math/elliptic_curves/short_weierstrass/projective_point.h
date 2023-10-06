#ifndef TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
#define TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/node/base/node_module.h"

namespace tachyon::node::math {

template <typename ProjectivePointTy,
          typename BaseField = typename ProjectivePointTy::BaseField,
          typename Curve = typename ProjectivePointTy::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddProjectivePoint(NodeModule& m, std::string_view name) {
  m.NewClass<ProjectivePointTy>(name)
      .template AddConstructor<>()
      .template AddConstructor<const BaseField&, const BaseField&,
                               const BaseField&>()
      .AddStaticMethod("zero", &ProjectivePointTy::Zero)
      .AddStaticMethod("generator", &ProjectivePointTy::Generator)
      .AddStaticMethod("random", &ProjectivePointTy::Random)
      .AddReadOnlyProperty("x", &ProjectivePointTy::x)
      .AddReadOnlyProperty("y", &ProjectivePointTy::y)
      .AddReadOnlyProperty("z", &ProjectivePointTy::z)
      .AddMethod("isZero", &ProjectivePointTy::IsZero)
      .AddMethod("isOnCurve", &ProjectivePointTy::IsOnCurve)
      .AddMethod("toString", &ProjectivePointTy::ToString)
      .AddMethod("toHexString", &ProjectivePointTy::ToHexString)
      .AddMethod("eq", &ProjectivePointTy::operator==)
      .AddMethod("ne", &ProjectivePointTy::operator!=)
      .AddMethod("add", &ProjectivePointTy::template operator+
                        <const ProjectivePointTy&>)
      .AddMethod("addMixed",
                 &ProjectivePointTy::template operator+<const AffinePointTy&>)
      .AddMethod("sub", &ProjectivePointTy::template operator-
                        <const ProjectivePointTy&>)
      .AddMethod("subMixed",
                 &ProjectivePointTy::template operator-<const AffinePointTy&>)
      .AddMethod("negative", &ProjectivePointTy::Negative)
      .AddMethod("double", &ProjectivePointTy::Double);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
