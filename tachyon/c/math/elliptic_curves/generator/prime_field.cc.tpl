// clang-format off
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}_traits.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/%{suffix}.h"

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_zero() {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  return c_cast(NativeType::Zero());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_one() {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  return c_cast(NativeType::One());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_random() {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  return c_cast(NativeType::Random());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_add(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a += native_cast(*b));
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_sub(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a -= native_cast(*b));
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_mul(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a *= native_cast(*b));
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_div(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a /= native_cast(*b));
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_neg(const tachyon_%{type}_%{suffix}* a) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.NegateInPlace());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_dbl(const tachyon_%{type}_%{suffix}* a) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.DoubleInPlace());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_sqr(const tachyon_%{type}_%{suffix}* a) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.SquareInPlace());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_inv(const tachyon_%{type}_%{suffix}* a) {
  using namespace tachyon::c::base;
  using NativeType = typename TypeTraits<tachyon_%{type}_%{suffix}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.InverseInPlace());
}

bool tachyon_%{type}_%{suffix}_eq(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) == native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_ne(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) != native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_gt(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) > native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_ge(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) >= native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_lt(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) < native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_le(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::c::base;
    return native_cast(*a) <= native_cast(*b);
}

// clang-format on
