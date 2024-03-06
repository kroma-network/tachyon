#include "tachyon/c/math/elliptic_curves/bn/bn254/fp2.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/extension_field_traits.h"
#include "tachyon/cc/math/finite_fields/extension_field_conversions.h"

tachyon_bn254_fp2 tachyon_bn254_fp2_zero() {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  return c_cast(ExtensionField::Zero());
}

tachyon_bn254_fp2 tachyon_bn254_fp2_one() {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  return c_cast(ExtensionField::One());
}

tachyon_bn254_fp2 tachyon_bn254_fp2_random() {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  return c_cast(ExtensionField::Random());
}

tachyon_bn254_fp2 tachyon_bn254_fp2_dbl(const tachyon_bn254_fp2* a) {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  ExtensionField native_a = native_cast(*a);
  return c_cast(native_a.DoubleInPlace());
}

tachyon_bn254_fp2 tachyon_bn254_fp2_neg(const tachyon_bn254_fp2* a) {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  ExtensionField native_a = native_cast(*a);
  return c_cast(native_a.NegInPlace());
}

tachyon_bn254_fp2 tachyon_bn254_fp2_sqr(const tachyon_bn254_fp2* a) {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  ExtensionField native_a = native_cast(*a);
  return c_cast(native_a.SquareInPlace());
}

tachyon_bn254_fp2 tachyon_bn254_fp2_inv(const tachyon_bn254_fp2* a) {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  ExtensionField native_a = native_cast(*a);
  return c_cast(native_a.InverseInPlace());
}

tachyon_bn254_fp2 tachyon_bn254_fp2_add(const tachyon_bn254_fp2* a,
                                        const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  ExtensionField native_a = native_cast(*a);
  return c_cast(native_a.AddInPlace(native_cast(*b)));
}

tachyon_bn254_fp2 tachyon_bn254_fp2_sub(const tachyon_bn254_fp2* a,
                                        const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  ExtensionField native_a = native_cast(*a);
  return c_cast(native_a.SubInPlace(native_cast(*b)));
}

tachyon_bn254_fp2 tachyon_bn254_fp2_mul(const tachyon_bn254_fp2* a,
                                        const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  ExtensionField native_a = native_cast(*a);
  return c_cast(native_a.MulInPlace(native_cast(*b)));
}

tachyon_bn254_fp2 tachyon_bn254_fp2_div(const tachyon_bn254_fp2* a,
                                        const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  using ExtensionField =
      typename ExtensionFieldTraits<tachyon_bn254_fp2>::ExtensionField;
  ExtensionField native_a = native_cast(*a);
  return c_cast(native_a.DivInPlace(native_cast(*b)));
}

bool tachyon_bn254_fp2_eq(const tachyon_bn254_fp2* a,
                          const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  return native_cast(*a) == native_cast(*b);
}

bool tachyon_bn254_fp2_ne(const tachyon_bn254_fp2* a,
                          const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  return native_cast(*a) != native_cast(*b);
}

bool tachyon_bn254_fp2_gt(const tachyon_bn254_fp2* a,
                          const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  return native_cast(*a) > native_cast(*b);
}

bool tachyon_bn254_fp2_ge(const tachyon_bn254_fp2* a,
                          const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  return native_cast(*a) >= native_cast(*b);
}

bool tachyon_bn254_fp2_lt(const tachyon_bn254_fp2* a,
                          const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  return native_cast(*a) < native_cast(*b);
}

bool tachyon_bn254_fp2_le(const tachyon_bn254_fp2* a,
                          const tachyon_bn254_fp2* b) {
  using namespace tachyon::cc::math;
  return native_cast(*a) <= native_cast(*b);
}
