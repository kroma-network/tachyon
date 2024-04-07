// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_CYCLOTOMIC_MULTIPLICATIVE_SUBGROUP_H_
#define TACHYON_MATH_FINITE_FIELDS_CYCLOTOMIC_MULTIPLICATIVE_SUBGROUP_H_

#include <tuple>
#include <vector>

#include "tachyon/base/containers/adapters.h"
#include "tachyon/math/finite_fields/finite_field.h"

namespace tachyon::math {
namespace internal {

SUPPORTS_UNARY_OPERATOR(FastCyclotomicSquare);
SUPPORTS_UNARY_OPERATOR(FastCyclotomicInverse);

}  // namespace internal

// clang-format off
// Fields that have a cyclotomic multiplicative subgroup, and which can
// leverage efficient inversion and squaring algorithms for elements in this
// subgroup. If a field has multiplicative order pᵈ - 1, the cyclotomic
// subgroups refer to subgroups of order φₙ(p), for any n < d, where φₙ is the
// [n-th cyclotomic polynomial](https://en.wikipedia.org/wiki/Cyclotomic_polynomial).
// clang-format on
template <typename F>
class CyclotomicMultiplicativeSubgroup : public FiniteField<F> {
 public:
  [[nodiscard]] constexpr F CyclotomicSquare() const {
    const F* f = static_cast<const F*>(this);
    if constexpr (internal::SupportsFastCyclotomicSquare<F>::value) {
      return f->FastCyclotomicSquare();
    } else {
      return f->Square();
    }
  }

  constexpr F& CyclotomicSquareInPlace() {
    F* f = static_cast<F*>(this);
    if constexpr (internal::SupportsFastCyclotomicSquareInPlace<F>::value) {
      return f->FastCyclotomicSquareInPlace();
    } else {
      return f->SquareInPlace();
    }
  }

  [[nodiscard]] constexpr F CyclotomicInverse() const {
    const F* f = static_cast<const F*>(this);
    if constexpr (internal::SupportsFastCyclotomicInverse<F>::value) {
      return f->FastCyclotomicInverse();
    } else {
      return f->Inverse();
    }
  }

  constexpr F& CyclotomicInverseInPlace() {
    F* f = static_cast<F*>(this);
    if constexpr (internal::SupportsFastCyclotomicInverseInPlace<F>::value) {
      return f->FastCyclotomicInverseInPlace();
    } else {
      return f->InverseInPlace();
    }
  }

  template <size_t N>
  [[nodiscard]] F CyclotomicPow(const BigInt<N>& exponent) const {
    const F* f = static_cast<const F*>(this);
    if (f->IsZero()) return F::Zero();

    if constexpr (internal::SupportsFastCyclotomicInverse<F>::value) {
      // We only use NAF-based exponentiation if inverses are fast to compute.
      std::vector<int8_t> naf = exponent.ToNAF();
      auto naf_rev_iterator = base::Reversed(naf);
      return DoCyclotomicPow(naf_rev_iterator.begin(), naf_rev_iterator.end());
    } else {
      return DoCyclotomicPow(BitIteratorBE<BigInt<N>>::begin(&exponent, true),
                             BitIteratorBE<BigInt<N>>::end(&exponent));
    }
  }

  template <size_t N>
  F& CyclotomicPowInPlace(const BigInt<N>& exponent) {
    F* f = static_cast<F*>(this);
    return *f = CyclotomicPow(exponent);
  }

 private:
  // Helper function to calculate the double-and-add loop for exponentiation.
  template <typename Iterator>
  constexpr F DoCyclotomicPow(Iterator begin, Iterator end) const {
    const F* f = static_cast<const F*>(this);
    // If the inverse is fast and we're using naf, we compute the inverse of the
    // base. Otherwise we do nothing with the variable, so we default it to one.
    F inverse = F::One();
    if constexpr (internal::SupportsFastCyclotomicInverse<F>::value) {
      inverse = CyclotomicInverse();
    } else {
      std::ignore = inverse;
    }

    F ret = F::One();
    bool found_nonzero = false;
    for (Iterator it = begin; it != end; ++it) {
      if (found_nonzero) {
        ret.CyclotomicSquareInPlace();
      }

      int8_t v = static_cast<int8_t>(*it);
      if (v == 0) continue;
      found_nonzero = true;

      if (v > 0) {
        ret *= *f;
      } else {
        if constexpr (internal::SupportsFastCyclotomicInverse<F>::value) {
          // only use naf if inversion is fast.
          ret *= inverse;
        }
      }
    }
    return ret;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_CYCLOTOMIC_MULTIPLICATIVE_SUBGROUP_H_
