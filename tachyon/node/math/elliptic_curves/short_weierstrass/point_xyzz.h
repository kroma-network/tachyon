#ifndef TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
#define TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/node/base/node_module.h"

namespace tachyon::node::math {

template <typename PointXYZZ,
          typename BaseField = typename PointXYZZ::BaseField,
          typename Curve = typename PointXYZZ::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddPointXYZZ(NodeModule& m, std::string_view name) {
  m.NewClass<PointXYZZ>(name)
      .template AddConstructor<>()
      .template AddConstructor<const BaseField&, const BaseField&,
                               const BaseField&, const BaseField&>()
      .AddStaticMethod("zero", &PointXYZZ::Zero)
      .AddStaticMethod("generator", &PointXYZZ::Generator)
      .AddStaticMethod("random", &PointXYZZ::Random)
      .AddReadOnlyProperty("x", &PointXYZZ::x)
      .AddReadOnlyProperty("y", &PointXYZZ::y)
      .AddReadOnlyProperty("zz", &PointXYZZ::zz)
      .AddReadOnlyProperty("zzz", &PointXYZZ::zzz)
      .AddMethod("isZero", &PointXYZZ::IsZero)
      .AddMethod("isOnCurve", &PointXYZZ::IsOnCurve)
      .AddMethod("toString", &PointXYZZ::ToString)
      .AddMethod("toHexString", &PointXYZZ::ToHexString, false)
      .AddMethod("eq", &PointXYZZ::operator==)
      .AddMethod("ne", &PointXYZZ::operator!=)
      .AddMethod("add", &PointXYZZ::template operator+ <const PointXYZZ&>)
      .AddMethod("addMixed",
                 &PointXYZZ::template operator+ <const AffinePointTy&>)
      .AddMethod("sub", &PointXYZZ::template operator- <const PointXYZZ&>)
      .AddMethod("subMixed",
                 &PointXYZZ::template operator- <const AffinePointTy&>)
      .AddMethod("negate", static_cast<PointXYZZ (PointXYZZ::*)() const>(
                               &PointXYZZ::operator-))
      .AddMethod("double", &PointXYZZ::Double);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
