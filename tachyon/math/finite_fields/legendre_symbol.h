#ifndef TACHYON_MATH_FINITE_FIELDS_LEGENDRE_SYMBOL_H_
#define TACHYON_MATH_FINITE_FIELDS_LEGENDRE_SYMBOL_H_

namespace tachyon::math {

// See https://en.wikipedia.org/wiki/Legendre_symbol
enum class LegendreSymbol {
  kOne = 1,
  kMinusOne = -1,
  kZero = 0,
  kQuadraticResidue = kOne,
  kNoneQuadraticResidue = kMinusOne,
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_LEGENDRE_SYMBOL_H_
