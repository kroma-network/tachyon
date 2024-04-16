// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_OPS_H_

#include <utility>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/math/polynomials/multivariate/multivariate_polynomial.h"

namespace tachyon::math {
namespace internal {

template <typename F, size_t MaxDegree>
class MultivariatePolynomialOp<MultivariateSparseCoefficients<F, MaxDegree>> {
 public:
  using S = MultivariateSparseCoefficients<F, MaxDegree>;
  using Term = typename S::Term;
  using Terms = std::vector<Term>;

  static MultivariatePolynomial<S> Add(const MultivariatePolynomial<S>& self,
                                       const MultivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return other;
    } else if (other.IsZero()) {
      return self;
    }

    MultivariatePolynomial<S> ret;
    DoAdd<false>(self, other, ret);
    return ret;
  }

  static MultivariatePolynomial<S>& AddInPlace(
      MultivariatePolynomial<S>& self, const MultivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return self = other;
    } else if (other.IsZero()) {
      return self;
    }

    DoAdd<false>(self, other, self);
    return self;
  }

  static MultivariatePolynomial<S> Sub(const MultivariatePolynomial<S>& self,
                                       const MultivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return -other;
    } else if (other.IsZero()) {
      return self;
    }

    MultivariatePolynomial<S> ret;
    DoAdd<true>(self, other, ret);
    return ret;
  }

  static MultivariatePolynomial<S>& SubInPlace(
      MultivariatePolynomial<S>& self, const MultivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return self = -other;
    } else if (other.IsZero()) {
      return self;
    }

    DoAdd<true>(self, other, self);
    return self;
  }

  static MultivariatePolynomial<S> Negative(
      const MultivariatePolynomial<S>& self) {
    if (self.IsZero()) {
      return self;
    }
    const Terms& i_terms = self.coefficients_.terms_;
    Terms o_terms(i_terms.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < o_terms.size(); ++i) {
      o_terms[i] = -i_terms[i];
    }
    return MultivariatePolynomial<S>(
        S(self.coefficients_.num_vars_, std::move(o_terms)));
  }

  static MultivariatePolynomial<S>& NegInPlace(
      MultivariatePolynomial<S>& self) {
    if (self.IsZero()) {
      return self;
    }
    Terms& terms = self.coefficients_.terms_;
    // clang-format off
    OPENMP_PARALLEL_FOR(Term& term : terms) { term.coefficient.NegInPlace(); }
    // clang-format on
    return self;
  }

 private:
  template <bool NEGATION>
  static void DoAdd(const MultivariatePolynomial<S>& a,
                    const MultivariatePolynomial<S>& b,
                    MultivariatePolynomial<S>& c) {
    const Terms& a_terms = a.coefficients_.terms_;
    const Terms& b_terms = b.coefficients_.terms_;
    Terms c_terms;

    auto a_it = a_terms.begin();
    auto b_it = b_terms.begin();
    while (a_it != a_terms.end() || b_it != b_terms.end()) {
      if (a_it == a_terms.end()) {
        if constexpr (NEGATION) {
          c_terms.push_back(-(*b_it));
        } else {
          c_terms.push_back(*b_it);
        }
        ++b_it;
        continue;
      }
      if (b_it == b_terms.end()) {
        c_terms.push_back(*a_it);
        ++a_it;
        continue;
      }
      if (a_it->literal < b_it->literal) {
        c_terms.push_back(*a_it);
        ++a_it;
      } else if (b_it->literal < a_it->literal) {
        if constexpr (NEGATION) {
          c_terms.push_back(-(*b_it));
        } else {
          c_terms.push_back(*b_it);
        }
        ++b_it;
      } else {
        F coeff;
        if constexpr (NEGATION) {
          coeff = a_it->coefficient - b_it->coefficient;
        } else {
          coeff = a_it->coefficient + b_it->coefficient;
        }
        if (!coeff.IsZero()) {
          c_terms.push_back({a_it->literal, std::move(coeff)});
        }
        ++a_it;
        ++b_it;
      }
    }

    c.coefficients_.terms_ = std::move(c_terms);
    c.coefficients_.Compact();
  }
};

}  // namespace internal
}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_OPS_H_
