#ifndef TACHYON_NODE_MATH_FINITE_FIELDS_PRIME_FIELD_H_
#define TACHYON_NODE_MATH_FINITE_FIELDS_PRIME_FIELD_H_

#include <string>

#include "tachyon/node/base/node_module.h"
#include "tachyon/node/math/base/big_int.h"

namespace tachyon::node::math {

template <typename PrimeFieldTy, size_t N>
PrimeFieldTy PrimeFieldFromNumber(const tachyon::math::BigInt<N>& value) {
  return PrimeFieldTy(value);
}

template <typename PrimeFieldTy>
PrimeFieldTy PrimeFieldFromDecString(const std::string& str) {
  return PrimeFieldTy::FromDecString(str);
}

template <typename PrimeFieldTy>
PrimeFieldTy PrimeFieldFromHexString(const std::string& str) {
  return PrimeFieldTy::FromHexString(str);
}

template <typename PrimeFieldTy, size_t N = PrimeFieldTy::N>
void AddPrimeField(NodeModule& m, std::string_view name) {
  m.NewClass<PrimeFieldTy>(name)
      .AddStaticMethod("zero", &PrimeFieldTy::Zero)
      .AddStaticMethod("one", &PrimeFieldTy::One)
      .AddStaticMethod("random", &PrimeFieldTy::Random)
      .AddStaticMethod("fromNumber", &PrimeFieldFromNumber<PrimeFieldTy, N>)
      .AddStaticMethod("fromDecString", &PrimeFieldFromDecString<PrimeFieldTy>)
      .AddStaticMethod("fromHexString", &PrimeFieldFromHexString<PrimeFieldTy>)
      .AddMethod("isZero", &PrimeFieldTy::IsZero)
      .AddMethod("isOne", &PrimeFieldTy::IsOne)
      .AddMethod("toString", &PrimeFieldTy::ToString)
      .AddMethod("toHexString", &PrimeFieldTy::ToHexString)
      .AddMethod("eq", &PrimeFieldTy::operator==)
      .AddMethod("ne", &PrimeFieldTy::operator!=)
      // NOLINTNEXTLINE(whitespace/operators)
      .AddMethod("lt", &PrimeFieldTy::operator<)
      .AddMethod("le", &PrimeFieldTy::operator<=)
      // NOLINTNEXTLINE(whitespace/operators)
      .AddMethod("gt", &PrimeFieldTy::operator>)
      .AddMethod("ge", &PrimeFieldTy::operator>=)
      .AddMethod("add", &PrimeFieldTy::template operator+ <const PrimeFieldTy&>)
      .AddMethod("sub", &PrimeFieldTy::template operator- <const PrimeFieldTy&>)
      .AddMethod("mul", &PrimeFieldTy::template operator* <const PrimeFieldTy&>)
      .AddMethod("div", &PrimeFieldTy::template operator/ <const PrimeFieldTy&>)
      .AddMethod("negative", &PrimeFieldTy::Negative)
      .AddMethod("double", &PrimeFieldTy::Double)
      .AddMethod("square", &PrimeFieldTy::Square);
}

}  // namespace tachyon::node::math

#endif  // TACHYON_NODE_MATH_FINITE_FIELDS_PRIME_FIELD_H_
