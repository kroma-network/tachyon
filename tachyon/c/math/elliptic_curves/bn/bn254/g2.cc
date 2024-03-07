#include "tachyon/c/math/elliptic_curves/bn/bn254/g2.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/extension_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_traits.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

void tachyon_bn254_g2_init() { tachyon::math::bn254::G2Curve::Init(); }

tachyon_bn254_g2_affine tachyon_bn254_g2_affine_zero() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_bn254_g2_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Zero());
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_zero() {
  using namespace tachyon::cc::math;
  using CurvePoint =
      typename PointTraits<tachyon_bn254_g2_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Zero());
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_zero() {
  using namespace tachyon::cc::math;
  using CurvePoint =
      typename PointTraits<tachyon_bn254_g2_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Zero());
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_zero() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_bn254_g2_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Zero());
}

tachyon_bn254_g2_affine tachyon_bn254_g2_affine_generator() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_bn254_g2_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Generator());
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_generator() {
  using namespace tachyon::cc::math;
  using CurvePoint =
      typename PointTraits<tachyon_bn254_g2_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Generator());
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_generator() {
  using namespace tachyon::cc::math;
  using CurvePoint =
      typename PointTraits<tachyon_bn254_g2_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Generator());
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_generator() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_bn254_g2_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Generator());
}

tachyon_bn254_g2_affine tachyon_bn254_g2_affine_random() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_bn254_g2_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Random());
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_random() {
  using namespace tachyon::cc::math;
  using CurvePoint =
      typename PointTraits<tachyon_bn254_g2_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Random());
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_random() {
  using namespace tachyon::cc::math;
  using CurvePoint =
      typename PointTraits<tachyon_bn254_g2_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Random());
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_random() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_bn254_g2_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Random());
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_affine_add(
    const tachyon_bn254_g2_affine* a, const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Add(ToAffinePoint(*b)));
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_add(
    const tachyon_bn254_g2_projective* a,
    const tachyon_bn254_g2_projective* b) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(
      ToProjectivePoint(*a).AddInPlace(ToProjectivePoint(*b)));
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_add_mixed(
    const tachyon_bn254_g2_projective* a, const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(
      ToProjectivePoint(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_add(
    const tachyon_bn254_g2_jacobian* a, const tachyon_bn254_g2_jacobian* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(ToJacobianPoint(*b)));
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_add_mixed(
    const tachyon_bn254_g2_jacobian* a, const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_add(
    const tachyon_bn254_g2_xyzz* a, const tachyon_bn254_g2_xyzz* b) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(ToPointXYZZ(*b)));
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_add_mixed(
    const tachyon_bn254_g2_xyzz* a, const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_affine_sub(
    const tachyon_bn254_g2_affine* a, const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Add(-ToAffinePoint(*b)));
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_sub(
    const tachyon_bn254_g2_projective* a,
    const tachyon_bn254_g2_projective* b) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(
      ToProjectivePoint(*a).AddInPlace(-ToProjectivePoint(*b)));
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_sub_mixed(
    const tachyon_bn254_g2_projective* a, const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(
      ToProjectivePoint(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_sub(
    const tachyon_bn254_g2_jacobian* a, const tachyon_bn254_g2_jacobian* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(-ToJacobianPoint(*b)));
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_sub_mixed(
    const tachyon_bn254_g2_jacobian* a, const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_sub(
    const tachyon_bn254_g2_xyzz* a, const tachyon_bn254_g2_xyzz* b) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(-ToPointXYZZ(*b)));
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_sub_mixed(
    const tachyon_bn254_g2_xyzz* a, const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_bn254_g2_affine tachyon_bn254_g2_affine_neg(
    const tachyon_bn254_g2_affine* a) {
  using namespace tachyon::cc::math;
  return ToCAffinePoint(ToAffinePoint(*a).NegInPlace());
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_neg(
    const tachyon_bn254_g2_projective* a) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).NegInPlace());
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_neg(
    const tachyon_bn254_g2_jacobian* a) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).NegInPlace());
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_neg(
    const tachyon_bn254_g2_xyzz* a) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).NegInPlace());
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_affine_dbl(
    const tachyon_bn254_g2_affine* a) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Double());
}

tachyon_bn254_g2_projective tachyon_bn254_g2_projective_dbl(
    const tachyon_bn254_g2_projective* a) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).DoubleInPlace());
}

tachyon_bn254_g2_jacobian tachyon_bn254_g2_jacobian_dbl(
    const tachyon_bn254_g2_jacobian* a) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).DoubleInPlace());
}

tachyon_bn254_g2_xyzz tachyon_bn254_g2_xyzz_dbl(
    const tachyon_bn254_g2_xyzz* a) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).DoubleInPlace());
}

bool tachyon_bn254_g2_affine_eq(const tachyon_bn254_g2_affine* a,
                                const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToAffinePoint(*a) == ToAffinePoint(*b);
}

bool tachyon_bn254_g2_projective_eq(const tachyon_bn254_g2_projective* a,
                                    const tachyon_bn254_g2_projective* b) {
  using namespace tachyon::cc::math;
  return ToProjectivePoint(*a) == ToProjectivePoint(*b);
}

bool tachyon_bn254_g2_jacobian_eq(const tachyon_bn254_g2_jacobian* a,
                                  const tachyon_bn254_g2_jacobian* b) {
  using namespace tachyon::cc::math;
  return ToJacobianPoint(*a) == ToJacobianPoint(*b);
}

bool tachyon_bn254_g2_xyzz_eq(const tachyon_bn254_g2_xyzz* a,
                              const tachyon_bn254_g2_xyzz* b) {
  using namespace tachyon::cc::math;
  return ToPointXYZZ(*a) == ToPointXYZZ(*b);
}

bool tachyon_bn254_g2_affine_ne(const tachyon_bn254_g2_affine* a,
                                const tachyon_bn254_g2_affine* b) {
  using namespace tachyon::cc::math;
  return ToAffinePoint(*a) != ToAffinePoint(*b);
}

bool tachyon_bn254_g2_projective_ne(const tachyon_bn254_g2_projective* a,
                                    const tachyon_bn254_g2_projective* b) {
  using namespace tachyon::cc::math;
  return ToProjectivePoint(*a) != ToProjectivePoint(*b);
}

bool tachyon_bn254_g2_jacobian_ne(const tachyon_bn254_g2_jacobian* a,
                                  const tachyon_bn254_g2_jacobian* b) {
  using namespace tachyon::cc::math;
  return ToJacobianPoint(*a) != ToJacobianPoint(*b);
}

bool tachyon_bn254_g2_xyzz_ne(const tachyon_bn254_g2_xyzz* a,
                              const tachyon_bn254_g2_xyzz* b) {
  using namespace tachyon::cc::math;
  return ToPointXYZZ(*a) != ToPointXYZZ(*b);
}
