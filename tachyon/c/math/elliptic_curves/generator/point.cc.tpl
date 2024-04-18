// clang-format off
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{base_field_trait_header}"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{g1_or_g2}_point_traits.h"
#include "tachyon/c/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/%{g1_or_g2}.h"

void tachyon_%{type}_%{g1_or_g2}_init() {
  tachyon::math::%{type}::%{G1_or_G2}Curve::Init();
}

tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_zero() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Zero());
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_zero() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Zero());
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_zero() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Zero());
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_zero() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Zero());
}

tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_generator() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Generator());
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_generator() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Generator());
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_generator() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Generator());
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_generator() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Generator());
}

tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_random() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Random());
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_random() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Random());
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_random() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Random());
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_random() {
  using namespace tachyon::c::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_%{g1_or_g2}_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Random());
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_add(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Add(ToAffinePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_add(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b) {
  using namespace tachyon::c::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).AddInPlace(ToProjectivePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_add_mixed(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_add(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(ToJacobianPoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_add_mixed(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_add(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b) {
  using namespace tachyon::c::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(ToPointXYZZ(*b)));
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_add_mixed(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_sub(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Add(-ToAffinePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_sub(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b) {
  using namespace tachyon::c::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).AddInPlace(-ToProjectivePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_sub(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(-ToJacobianPoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_sub(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b) {
  using namespace tachyon::c::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(-ToPointXYZZ(*b)));
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_neg(const tachyon_%{type}_%{g1_or_g2}_affine* a) {
  using namespace tachyon::c::math;
  return ToCAffinePoint(ToAffinePoint(*a).NegateInPlace());
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_neg(const tachyon_%{type}_%{g1_or_g2}_projective* a) {
  using namespace tachyon::c::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).NegateInPlace());
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_neg(const tachyon_%{type}_%{g1_or_g2}_jacobian* a) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).NegateInPlace());
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_neg(const tachyon_%{type}_%{g1_or_g2}_xyzz* a) {
  using namespace tachyon::c::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).NegateInPlace());
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_dbl(const tachyon_%{type}_%{g1_or_g2}_affine* a) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Double());
}

tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_dbl(const tachyon_%{type}_%{g1_or_g2}_projective* a) {
  using namespace tachyon::c::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).DoubleInPlace());
}

tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_dbl(const tachyon_%{type}_%{g1_or_g2}_jacobian* a) {
  using namespace tachyon::c::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).DoubleInPlace());
}

tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_dbl(const tachyon_%{type}_%{g1_or_g2}_xyzz* a) {
  using namespace tachyon::c::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).DoubleInPlace());
}

bool tachyon_%{type}_%{g1_or_g2}_affine_eq(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToAffinePoint(*a) == ToAffinePoint(*b);
}

bool tachyon_%{type}_%{g1_or_g2}_projective_eq(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b) {
  using namespace tachyon::c::math;
  return ToProjectivePoint(*a) == ToProjectivePoint(*b);
}

bool tachyon_%{type}_%{g1_or_g2}_jacobian_eq(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b) {
  using namespace tachyon::c::math;
  return ToJacobianPoint(*a) == ToJacobianPoint(*b);
}

bool tachyon_%{type}_%{g1_or_g2}_xyzz_eq(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b) {
  using namespace tachyon::c::math;
  return ToPointXYZZ(*a) == ToPointXYZZ(*b);
}

bool tachyon_%{type}_%{g1_or_g2}_affine_ne(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b) {
  using namespace tachyon::c::math;
  return ToAffinePoint(*a) != ToAffinePoint(*b);
}

bool tachyon_%{type}_%{g1_or_g2}_projective_ne(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b) {
  using namespace tachyon::c::math;
  return ToProjectivePoint(*a) != ToProjectivePoint(*b);
}

bool tachyon_%{type}_%{g1_or_g2}_jacobian_ne(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b) {
  using namespace tachyon::c::math;
  return ToJacobianPoint(*a) != ToJacobianPoint(*b);
}

bool tachyon_%{type}_%{g1_or_g2}_xyzz_ne(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b) {
  using namespace tachyon::c::math;
  return ToPointXYZZ(*a) != ToPointXYZZ(*b);
}
// clang-format off
