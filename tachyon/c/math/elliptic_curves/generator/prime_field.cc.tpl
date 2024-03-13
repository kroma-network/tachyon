// clang-format off
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}_prime_field_traits.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/%{suffix}.h"

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_zero() {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  return c_cast(PrimeField::Zero());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_one() {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  return c_cast(PrimeField::One());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_random() {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  return c_cast(PrimeField::Random());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_add(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  PrimeField native_a = native_cast(*a);
  return c_cast(native_a.AddInPlace(native_cast(*b)));
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_sub(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  PrimeField native_a = native_cast(*a);
  return c_cast(native_a.SubInPlace(native_cast(*b)));
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_mul(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  PrimeField native_a = native_cast(*a);
  return c_cast(native_a.MulInPlace(native_cast(*b)));
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_div(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  PrimeField native_a = native_cast(*a);
  return c_cast(native_a.DivInPlace(native_cast(*b)));
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_neg(const tachyon_%{type}_%{suffix}* a) {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  PrimeField native_a = native_cast(*a);
  return c_cast(native_a.NegInPlace());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_dbl(const tachyon_%{type}_%{suffix}* a) {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  PrimeField native_a = native_cast(*a);
  return c_cast(native_a.DoubleInPlace());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_sqr(const tachyon_%{type}_%{suffix}* a) {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  PrimeField native_a = native_cast(*a);
  return c_cast(native_a.SquareInPlace());
}

tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_inv(const tachyon_%{type}_%{suffix}* a) {
  using namespace tachyon::cc::math;
  using PrimeField = typename PrimeFieldTraits<tachyon_%{type}_%{suffix}>::PrimeField;
  PrimeField native_a = native_cast(*a);
  return c_cast(native_a.InverseInPlace());
}

bool tachyon_%{type}_%{suffix}_eq(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::cc::math;
    return native_cast(*a) == native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_ne(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::cc::math;
    return native_cast(*a) != native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_gt(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::cc::math;
    return native_cast(*a) > native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_ge(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::cc::math;
    return native_cast(*a) >= native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_lt(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::cc::math;
    return native_cast(*a) < native_cast(*b);
}

bool tachyon_%{type}_%{suffix}_le(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b) {
    using namespace tachyon::cc::math;
    return native_cast(*a) <= native_cast(*b);
}

// clang-format on
