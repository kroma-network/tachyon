#ifndef TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
#define TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/node/base/node_module.h"

namespace tachyon::node::math {

template <typename PointXYZZTy,
          typename BaseField = typename PointXYZZTy::BaseField,
          typename Curve = typename PointXYZZTy::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddPointXYZZ(NodeModule& m, std::string_view name) {
  m.NewClass<PointXYZZTy>(name)
      .template AddConstructor<>()
      .template AddConstructor<const BaseField&, const BaseField&,
                               const BaseField&, const BaseField&>()
      .AddStaticMethod("zero", &PointXYZZTy::Zero)
      .AddStaticMethod("generator", &PointXYZZTy::Generator)
      .AddStaticMethod("random", &PointXYZZTy::Random)
      .AddReadOnlyProperty("x", &PointXYZZTy::x)
      .AddReadOnlyProperty("y", &PointXYZZTy::y)
      .AddReadOnlyProperty("zz", &PointXYZZTy::zz)
      .AddReadOnlyProperty("zzz", &PointXYZZTy::zzz)
      .AddMethod("isZero", &PointXYZZTy::IsZero)
      .AddMethod("isOnCurve", &PointXYZZTy::IsOnCurve)
      .AddMethod("toString", &PointXYZZTy::ToString)
      .AddMethod("toHexString", &PointXYZZTy::ToHexString)
      .AddMethod("eq", &PointXYZZTy::operator==)
      .AddMethod("ne", &PointXYZZTy::operator!=)
      .AddMethod("add",
                 &PointXYZZTy::template operator+<const PointXYZZTy&>)
      .AddMethod("addMixed",
                 &PointXYZZTy::template operator+<const AffinePointTy&>)
      .AddMethod("sub",
                 &PointXYZZTy::template operator-<const PointXYZZTy&>)
      .AddMethod("subMixed",
                 &PointXYZZTy::template operator-<const AffinePointTy&>)
      .AddMethod("negative", &PointXYZZTy::Negative)
      .AddMethod("double", &PointXYZZTy::Double);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
