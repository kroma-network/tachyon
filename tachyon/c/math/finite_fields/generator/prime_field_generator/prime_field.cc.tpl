// clang-format off
#include "tachyon/c/base/type_traits_forward.h"
#include "%{c_type_traits_hdr}"

tachyon_%{class_name} tachyon_%{class_name}_zero() {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  return c_cast(NativeType::Zero());
}

tachyon_%{class_name} tachyon_%{class_name}_one() {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  return c_cast(NativeType::One());
}

tachyon_%{class_name} tachyon_%{class_name}_minus_one() {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  return c_cast(NativeType::MinusOne());
}

tachyon_%{class_name} tachyon_%{class_name}_random() {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  return c_cast(NativeType::Random());
}

tachyon_%{class_name} tachyon_%{class_name}_add(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a += native_cast(*b));
}

tachyon_%{class_name} tachyon_%{class_name}_sub(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a -= native_cast(*b));
}

tachyon_%{class_name} tachyon_%{class_name}_mul(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a *= native_cast(*b));
}

tachyon_%{class_name} tachyon_%{class_name}_div(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  NativeType native_a = native_cast(*a);
  CHECK(native_a /= native_cast(*b));
  return c_cast(native_a);
}

tachyon_%{class_name} tachyon_%{class_name}_neg(const tachyon_%{class_name}* a) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.NegateInPlace());
}

tachyon_%{class_name} tachyon_%{class_name}_dbl(const tachyon_%{class_name}* a) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.DoubleInPlace());
}

tachyon_%{class_name} tachyon_%{class_name}_sqr(const tachyon_%{class_name}* a) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.SquareInPlace());
}

tachyon_%{class_name} tachyon_%{class_name}_inv(const tachyon_%{class_name}* a) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{class_name}>::NativeType;
  NativeType native_a = native_cast(*a);
  CHECK(native_a.InverseInPlace());
  return c_cast(native_a);
}

bool tachyon_%{class_name}_eq(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) == native_cast(*b);
}

bool tachyon_%{class_name}_ne(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) != native_cast(*b);
}

bool tachyon_%{class_name}_gt(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) > native_cast(*b);
}

bool tachyon_%{class_name}_ge(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) >= native_cast(*b);
}

bool tachyon_%{class_name}_lt(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) < native_cast(*b);
}

bool tachyon_%{class_name}_le(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) <= native_cast(*b);
}

// clang-format on
