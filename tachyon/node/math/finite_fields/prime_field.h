#ifndef TACHYON_NODE_MATH_FINITE_FIELDS_PRIME_FIELD_H_
#define TACHYON_NODE_MATH_FINITE_FIELDS_PRIME_FIELD_H_

#include <string>

#include "tachyon/node/base/node_module.h"
#include "tachyon/node/math/base/big_int.h"

namespace tachyon::node::math {

template <typename PrimeField, size_t N>
PrimeField PrimeFieldFromNumber(const tachyon::math::BigInt<N>& value) {
  return PrimeField(value);
}

template <typename PrimeField>
PrimeField PrimeFieldFromDecString(const std::string& str) {
  // TODO(chokobole): Throw errors
  return *PrimeField::FromDecString(str);
}

template <typename PrimeField>
PrimeField PrimeFieldFromHexString(const std::string& str) {
  // TODO(chokobole): Throw errors
  return *PrimeField::FromHexString(str);
}

template <typename PrimeField, size_t N = PrimeField::N>
void AddPrimeField(NodeModule& m, std::string_view name) {
  m.NewClass<PrimeField>(name)
      .AddStaticMethod("zero", &PrimeField::Zero)
      .AddStaticMethod("one", &PrimeField::One)
      .AddStaticMethod("random", &PrimeField::Random)
      .AddStaticMethod("fromNumber", &PrimeFieldFromNumber<PrimeField, N>)
      .AddStaticMethod("fromDecString", &PrimeFieldFromDecString<PrimeField>)
      .AddStaticMethod("fromHexString", &PrimeFieldFromHexString<PrimeField>)
      .AddMethod("isZero", &PrimeField::IsZero)
      .AddMethod("isOne", &PrimeField::IsOne)
      .AddMethod("toString", &PrimeField::ToString)
      .AddMethod("toHexString", &PrimeField::ToHexString, false)
      .AddMethod("eq", &PrimeField::operator==)
      .AddMethod("ne", &PrimeField::operator!=)
      // NOLINTNEXTLINE(whitespace/operators)
      .AddMethod("lt", &PrimeField::operator<)
      .AddMethod("le", &PrimeField::operator<=)
      // NOLINTNEXTLINE(whitespace/operators)
      .AddMethod("gt", &PrimeField::operator>)
      .AddMethod("ge", &PrimeField::operator>=)
      .AddMethod("add", &PrimeField::template operator+ <const PrimeField&>)
      .AddMethod("sub", &PrimeField::template operator- <const PrimeField&>)
      .AddMethod("mul", &PrimeField::template operator* <const PrimeField&>)
      .AddMethod("div", &PrimeField::template operator/ <const PrimeField&>)
      .AddMethod("negate", static_cast<PrimeField (PrimeField::*)() const>(
                               &PrimeField::operator-))
      .AddMethod("double", &PrimeField::Double)
      .AddMethod("square", &PrimeField::Square);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_FINITE_FIELDS_PRIME_FIELD_H_
