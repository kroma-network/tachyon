// clang-format off
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_traits.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/g1.h"

void tachyon_%{type}_g1_init() {
  tachyon::math::%{type}::G1Curve::Init();
}

tachyon_%{type}_g1_affine tachyon_%{type}_g1_affine_zero() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Zero());
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_zero() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Zero());
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_zero() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Zero());
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_zero() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Zero());
}

tachyon_%{type}_g1_affine tachyon_%{type}_g1_affine_generator() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Generator());
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_generator() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Generator());
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_generator() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Generator());
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_generator() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Generator());
}

tachyon_%{type}_g1_affine tachyon_%{type}_g1_affine_random() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_affine>::CurvePoint;
  return ToCAffinePoint(CurvePoint::Random());
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_random() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_projective>::CurvePoint;
  return ToCProjectivePoint(CurvePoint::Random());
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_random() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_jacobian>::CurvePoint;
  return ToCJacobianPoint(CurvePoint::Random());
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_random() {
  using namespace tachyon::cc::math;
  using CurvePoint = typename PointTraits<tachyon_%{type}_g1_xyzz>::CurvePoint;
  return ToCPointXYZZ(CurvePoint::Random());
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_affine_add(const tachyon_%{type}_g1_affine* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Add(ToAffinePoint(*b)));
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_add(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_projective* b) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).AddInPlace(ToProjectivePoint(*b)));
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_add_mixed(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_add(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_jacobian* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(ToJacobianPoint(*b)));
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_add_mixed(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_add(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_xyzz* b) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(ToPointXYZZ(*b)));
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_add_mixed(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(ToAffinePoint(*b)));
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_affine_sub(const tachyon_%{type}_g1_affine* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Add(-ToAffinePoint(*b)));
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_sub(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_projective* b) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).AddInPlace(-ToProjectivePoint(*b)));
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_sub_mixed(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_sub(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_jacobian* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(-ToJacobianPoint(*b)));
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_sub_mixed(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_sub(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_xyzz* b) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(-ToPointXYZZ(*b)));
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_sub_mixed(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).AddInPlace(-ToAffinePoint(*b)));
}

tachyon_%{type}_g1_affine tachyon_%{type}_g1_affine_neg(const tachyon_%{type}_g1_affine* a) {
  using namespace tachyon::cc::math;
  return ToCAffinePoint(ToAffinePoint(*a).NegInPlace());
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_neg(const tachyon_%{type}_g1_projective* a) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).NegInPlace());
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_neg(const tachyon_%{type}_g1_jacobian* a) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).NegInPlace());
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_neg(const tachyon_%{type}_g1_xyzz* a) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).NegInPlace());
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_affine_dbl(const tachyon_%{type}_g1_affine* a) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToAffinePoint(*a).Double());
}

tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_dbl(const tachyon_%{type}_g1_projective* a) {
  using namespace tachyon::cc::math;
  return ToCProjectivePoint(ToProjectivePoint(*a).DoubleInPlace());
}

tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_dbl(const tachyon_%{type}_g1_jacobian* a) {
  using namespace tachyon::cc::math;
  return ToCJacobianPoint(ToJacobianPoint(*a).DoubleInPlace());
}

tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_dbl(const tachyon_%{type}_g1_xyzz* a) {
  using namespace tachyon::cc::math;
  return ToCPointXYZZ(ToPointXYZZ(*a).DoubleInPlace());
}

bool tachyon_%{type}_g1_affine_eq(const tachyon_%{type}_g1_affine* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToAffinePoint(*a) == ToAffinePoint(*b);
}

bool tachyon_%{type}_g1_projective_eq(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_projective* b) {
  using namespace tachyon::cc::math;
  return ToProjectivePoint(*a) == ToProjectivePoint(*b);
}

bool tachyon_%{type}_g1_jacobian_eq(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_jacobian* b) {
  using namespace tachyon::cc::math;
  return ToJacobianPoint(*a) == ToJacobianPoint(*b);
}

bool tachyon_%{type}_g1_xyzz_eq(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_xyzz* b) {
  using namespace tachyon::cc::math;
  return ToPointXYZZ(*a) == ToPointXYZZ(*b);
}

bool tachyon_%{type}_g1_affine_ne(const tachyon_%{type}_g1_affine* a, const tachyon_%{type}_g1_affine* b) {
  using namespace tachyon::cc::math;
  return ToAffinePoint(*a) != ToAffinePoint(*b);
}

bool tachyon_%{type}_g1_projective_ne(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_projective* b) {
  using namespace tachyon::cc::math;
  return ToProjectivePoint(*a) != ToProjectivePoint(*b);
}

bool tachyon_%{type}_g1_jacobian_ne(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_jacobian* b) {
  using namespace tachyon::cc::math;
  return ToJacobianPoint(*a) != ToJacobianPoint(*b);
}

bool tachyon_%{type}_g1_xyzz_ne(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_xyzz* b) {
  using namespace tachyon::cc::math;
  return ToPointXYZZ(*a) != ToPointXYZZ(*b);
}
// clang-format off
