// clang-format off
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fq%{degree}_type_traits.h"

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_zero() {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  return c_cast(NativeType::Zero());
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_one() {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  return c_cast(NativeType::One());
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_random() {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  return c_cast(NativeType::Random());
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_dbl(const tachyon_%{type}_fq%{degree}* a) {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.DoubleInPlace());
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_neg(const tachyon_%{type}_fq%{degree}* a) {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.NegateInPlace());
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_sqr(const tachyon_%{type}_fq%{degree}* a) {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a.SquareInPlace());
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_inv(const tachyon_%{type}_fq%{degree}* a) {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  NativeType native_a = native_cast(*a);
  CHECK(native_a.InverseInPlace());
  return c_cast(native_a);
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_add(const tachyon_%{type}_fq%{degree}* a,
                                        const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a += native_cast(*b));
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_sub(const tachyon_%{type}_fq%{degree}* a,
                                        const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a -= native_cast(*b));
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_mul(const tachyon_%{type}_fq%{degree}* a,
                                        const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  NativeType native_a = native_cast(*a);
  return c_cast(native_a *= native_cast(*b));
}

tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_div(const tachyon_%{type}_fq%{degree}* a,
                                        const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  using NativeType =
      typename TypeTraits<tachyon_%{type}_fq%{degree}>::NativeType;
  NativeType native_a = native_cast(*a);
  CHECK(native_a /= native_cast(*b));
  return c_cast(native_a);
}

bool tachyon_%{type}_fq%{degree}_eq(const tachyon_%{type}_fq%{degree}* a,
                          const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  return native_cast(*a) == native_cast(*b);
}

bool tachyon_%{type}_fq%{degree}_ne(const tachyon_%{type}_fq%{degree}* a,
                          const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  return native_cast(*a) != native_cast(*b);
}

bool tachyon_%{type}_fq%{degree}_gt(const tachyon_%{type}_fq%{degree}* a,
                          const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  return native_cast(*a) > native_cast(*b);
}

bool tachyon_%{type}_fq%{degree}_ge(const tachyon_%{type}_fq%{degree}* a,
                          const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  return native_cast(*a) >= native_cast(*b);
}

bool tachyon_%{type}_fq%{degree}_lt(const tachyon_%{type}_fq%{degree}* a,
                          const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  return native_cast(*a) < native_cast(*b);
}

bool tachyon_%{type}_fq%{degree}_le(const tachyon_%{type}_fq%{degree}* a,
                          const tachyon_%{type}_fq%{degree}* b) {
  using namespace tachyon::c::base;
  return native_cast(*a) <= native_cast(*b);
}
// clang-format on
