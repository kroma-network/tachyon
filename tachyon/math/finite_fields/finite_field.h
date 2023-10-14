#ifndef TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_H_

#include "tachyon/math/base/field.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"
#include "tachyon/math/finite_fields/square_root_algorithms/shanks.h"
#include "tachyon/math/finite_fields/square_root_algorithms/tonelli_shanks.h"

namespace tachyon::math {

// FiniteField is a field with a finite field order (i.e., number of
// elements), also called a Galois field. The order of a finite field is
// always a prime or a power of a prime (Birkhoff and Mac Lane 1996). For each
// prime power, there exists exactly one (with the usual caveat that "exactly
// one" means "exactly one up to an isomorphism") finite field GF(pⁿ), often
// written as F(pⁿ) in current usage.
// See https://mathworld.wolfram.com/FiniteField.html
template <typename F>
class FiniteField : public Field<F> {
 public:
  using Config = typename FiniteFieldTraits<F>::Config;

  constexpr bool SquareRoot(F* ret) const {
    if constexpr (Config::kModulusModFourIsThree) {
      return ComputeShanksSquareRoot(*static_cast<const F*>(this), ret);
    } else {
      static_assert(Config::kHasTwoAdicRootOfUnity);
      return ComputeTonelliShanksSquareRoot(
          *static_cast<const F*>(this),
          F::FromMontgomery(Config::kTwoAdicRootOfUnity), ret);
    }
    return false;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_H_
