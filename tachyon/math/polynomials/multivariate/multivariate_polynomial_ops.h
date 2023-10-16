// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_OPS_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
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

  static MultivariatePolynomial<S>& AddInPlace(
      MultivariatePolynomial<S>& self, const MultivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return self = other;
    } else if (other.IsZero()) {
      return self;
    }

    return DoAddition<false>(self, other);
  }

  static MultivariatePolynomial<S>& SubInPlace(
      MultivariatePolynomial<S>& self, const MultivariatePolynomial<S>& other) {
    Terms& l_terms = self.coefficients_.terms_;
    const Terms& r_terms = other.coefficients_.terms_;
    if (self.IsZero()) {
      l_terms = base::CreateVector(
          r_terms.size(), [&r_terms](size_t idx) { return -r_terms[idx]; });
      return self;
    } else if (other.IsZero()) {
      return self;
    }

    return DoAddition<true>(self, other);
  }

  static MultivariatePolynomial<S>& NegInPlace(
      MultivariatePolynomial<S>& self) {
    Terms& terms = self.terms_;
    // clang-format off
    OPENMP_PARALLEL_FOR(Term& term : terms) { term.coefficient.NegInPlace(); }
    // clang-format on
    return self;
  }

 private:
  template <bool NEGATION>
  static MultivariatePolynomial<S>& DoAddition(
      MultivariatePolynomial<S>& self, const MultivariatePolynomial<S>& other) {
    Terms& l_terms = self.coefficients_.terms_;
    const Terms& r_terms = other.coefficients_.terms_;

    auto l_it = l_terms.begin();
    auto r_it = r_terms.begin();
    Terms ret;
    while (l_it != l_terms.end() || r_it != r_terms.end()) {
      if (l_it == l_terms.end()) {
        if constexpr (NEGATION) {
          ret.push_back(-(*r_it));
        } else {
          ret.push_back(*r_it);
        }
        ++r_it;
        continue;
      }
      if (r_it == r_terms.end()) {
        ret.push_back(*l_it);
        ++l_it;
        continue;
      }
      if (l_it->literal < r_it->literal) {
        ret.push_back(*l_it);
        ++l_it;
      } else if (r_it->literal < l_it->literal) {
        if constexpr (NEGATION) {
          ret.push_back(-(*r_it));
        } else {
          ret.push_back(*r_it);
        }
        ++r_it;
      } else {
        F coeff;
        if constexpr (NEGATION) {
          coeff = l_it->coefficient - r_it->coefficient;
        } else {
          coeff = l_it->coefficient + r_it->coefficient;
        }
        if (!coeff.IsZero()) {
          ret.push_back({l_it->literal, std::move(coeff)});
        }
        ++l_it;
        ++r_it;
      }
    }

    l_terms = std::move(ret);
    self.coefficients_.Compact();
    return self;
  }
};

}  // namespace internal
}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_OPS_H_
