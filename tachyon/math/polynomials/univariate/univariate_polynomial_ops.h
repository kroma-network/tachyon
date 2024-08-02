// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_OPS_H_

#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "third_party/pdqsort/include/pdqsort.h"

#include "tachyon/base/memory/reusing_allocator.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/optional.h"
#include "tachyon/math/base/arithmetics_results.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {
namespace internal {

template <typename F, size_t MaxDegree>
class UnivariatePolynomialOp<UnivariateDenseCoefficients<F, MaxDegree>> {
 public:
  using D = UnivariateDenseCoefficients<F, MaxDegree>;
  using S = UnivariateSparseCoefficients<F, MaxDegree>;
  using Term = typename S::Term;

  static UnivariatePolynomial<D> Add(const UnivariatePolynomial<D>& self,
                                     const UnivariatePolynomial<D>& other) {
    if (self.IsZero()) {
      return other;
    } else if (other.IsZero()) {
      return self;
    }

    const std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    const std::vector<F, base::memory::ReusingAllocator<F>>& r_coefficients =
        other.coefficients_.coefficients_;
    UnivariatePolynomial<D> ret;
    std::vector<F, base::memory::ReusingAllocator<F>>& o_coefficients =
        ret.coefficients_.coefficients_;
    o_coefficients.resize(
        std::max(l_coefficients.size(), r_coefficients.size()));
    OPENMP_PARALLEL_FOR(size_t i = 0; i < o_coefficients.size(); ++i) {
      o_coefficients[i] = self.coefficients_[i] + other.coefficients_[i];
    }

    ret.coefficients_.RemoveHighDegreeZeros();
    return ret;
  }

  static UnivariatePolynomial<D>& AddInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<D>& other) {
    if (self.IsZero()) {
      return self = other;
    } else if (other.IsZero()) {
      return self;
    }

    std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    const std::vector<F, base::memory::ReusingAllocator<F>>& r_coefficients =
        other.coefficients_.coefficients_;
    l_coefficients.resize(
        std::max(l_coefficients.size(), r_coefficients.size()));
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_coefficients.size(); ++i) {
      l_coefficients[i] += r_coefficients[i];
    }

    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D> Add(const UnivariatePolynomial<D>& self,
                                     const UnivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return other.ToDense();
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    UnivariatePolynomial<D> ret;
    std::vector<F, base::memory::ReusingAllocator<F>>& o_coefficients =
        ret.coefficients_.coefficients_;
    o_coefficients.resize(std::max(degree, other_degree) + 1);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < o_coefficients.size(); ++i) {
      o_coefficients[i] = self.coefficients_[i] + other.coefficients()[i];
    }

    ret.coefficients_.RemoveHighDegreeZeros();
    return ret;
  }

  static UnivariatePolynomial<D>& AddInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return Copy<false>(self, other);
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    std::vector<F, base::memory::ReusingAllocator<F>> upper_coeffs;
    if (degree < other_degree) {
      upper_coeffs = std::vector<F, base::memory::ReusingAllocator<F>>(
          other_degree - degree);
    }

    std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    const std::vector<Term>& r_terms = other.coefficients().terms_;
    OPENMP_PARALLEL_FOR(const Term& r_term : r_terms) {
      if (r_term.degree <= degree) {
        l_coefficients[r_term.degree] += r_term.coefficient;
      } else {
        upper_coeffs[r_term.degree - degree - 1] = r_term.coefficient;
      }
    }
    l_coefficients.insert(l_coefficients.end(),
                          std::make_move_iterator(upper_coeffs.begin()),
                          std::make_move_iterator(upper_coeffs.end()));

    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D> Sub(const UnivariatePolynomial<D>& self,
                                     const UnivariatePolynomial<D>& other) {
    if (self.IsZero()) {
      return -other;
    } else if (other.IsZero()) {
      return self;
    }

    const std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    const std::vector<F, base::memory::ReusingAllocator<F>>& r_coefficients =
        other.coefficients_.coefficients_;
    UnivariatePolynomial<D> ret;
    std::vector<F, base::memory::ReusingAllocator<F>>& o_coefficients =
        ret.coefficients_.coefficients_;
    o_coefficients.resize(
        std::max(l_coefficients.size(), r_coefficients.size()));
    OPENMP_PARALLEL_FOR(size_t i = 0; i < o_coefficients.size(); ++i) {
      o_coefficients[i] = self.coefficients_[i] - other.coefficients_[i];
    }

    ret.coefficients_.RemoveHighDegreeZeros();
    return ret;
  }

  static UnivariatePolynomial<D>& SubInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<D>& other) {
    if (self.IsZero()) {
      return self = -other;
    } else if (other.IsZero()) {
      return self;
    }

    std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    const std::vector<F, base::memory::ReusingAllocator<F>>& r_coefficients =
        other.coefficients_.coefficients_;
    l_coefficients.resize(
        std::max(l_coefficients.size(), r_coefficients.size()));
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_coefficients.size(); ++i) {
      l_coefficients[i] -= r_coefficients[i];
    }

    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D> Sub(const UnivariatePolynomial<D>& self,
                                     const UnivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return (-other).ToDense();
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    UnivariatePolynomial<D> ret;
    std::vector<F, base::memory::ReusingAllocator<F>>& o_coefficients =
        ret.coefficients_.coefficients_;
    o_coefficients.resize(std::max(degree, other_degree) + 1);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < o_coefficients.size(); ++i) {
      o_coefficients[i] = self.coefficients_[i] - other.coefficients()[i];
    }

    ret.coefficients_.RemoveHighDegreeZeros();
    return ret;
  }

  static UnivariatePolynomial<D>& SubInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return Copy<true>(self, other);
    } else if (other.IsZero()) {
      return self;
    }

    size_t degree = self.Degree();
    size_t other_degree = other.Degree();
    std::vector<F, base::memory::ReusingAllocator<F>> upper_coeffs;
    if (degree < other_degree) {
      upper_coeffs = std::vector<F, base::memory::ReusingAllocator<F>>(
          other_degree - degree);
    }

    std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    const std::vector<Term>& r_terms = other.coefficients().terms_;
    OPENMP_PARALLEL_FOR(const Term& r_term : r_terms) {
      if (r_term.degree <= degree) {
        l_coefficients[r_term.degree] -= r_term.coefficient;
      } else {
        upper_coeffs[r_term.degree - degree - 1] = -r_term.coefficient;
      }
    }
    l_coefficients.insert(l_coefficients.end(),
                          std::make_move_iterator(upper_coeffs.begin()),
                          std::make_move_iterator(upper_coeffs.end()));

    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D> Negate(const UnivariatePolynomial<D>& self) {
    if (self.IsZero()) {
      return self;
    }
    const std::vector<F, base::memory::ReusingAllocator<F>>& i_coefficients =
        self.coefficients_.coefficients_;
    std::vector<F, base::memory::ReusingAllocator<F>> o_coefficients(
        i_coefficients.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < o_coefficients.size(); ++i) {
      o_coefficients[i] = -i_coefficients[i];
    }
    return UnivariatePolynomial<D>(D(std::move(o_coefficients)));
  }

  static UnivariatePolynomial<D>& NegateInPlace(UnivariatePolynomial<D>& self) {
    if (self.IsZero()) {
      return self;
    }
    std::vector<F, base::memory::ReusingAllocator<F>>& coefficients =
        self.coefficients_.coefficients_;
    // clang-format off
    OPENMP_PARALLEL_FOR(F& coefficient : coefficients) {
      // clang-format on
      coefficient.NegateInPlace();
    }
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D> Mul(const UnivariatePolynomial<D>& self,
                                     const F& scalar) {
    if (self.IsZero() || scalar.IsOne()) {
      return self;
    } else if (scalar.IsZero()) {
      return UnivariatePolynomial<D>::Zero();
    }
    const std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    std::vector<F, base::memory::ReusingAllocator<F>> o_coefficients(
        l_coefficients.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_coefficients.size(); ++i) {
      o_coefficients[i] = l_coefficients[i] * scalar;
    }
    return UnivariatePolynomial<D>(D(std::move(o_coefficients)));
  }

  static UnivariatePolynomial<D>& MulInPlace(UnivariatePolynomial<D>& self,
                                             const F& scalar) {
    if (self.IsZero() || scalar.IsOne()) {
      return self;
    }
    std::vector<F, base::memory::ReusingAllocator<F>>& coefficients =
        self.coefficients_.coefficients_;
    // clang-format off
    OPENMP_PARALLEL_FOR(F& coefficient : coefficients) {
      // clang-format on
      coefficient *= scalar;
    }
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static UnivariatePolynomial<D> Mul(const UnivariatePolynomial<D>& self,
                                     const UnivariatePolynomial<D>& other) {
    if (self.IsZero() || other.IsZero()) {
      return UnivariatePolynomial<D>::Zero();
    } else if (self.IsOne()) {
      return other;
    } else if (other.IsOne()) {
      return self;
    }

    UnivariatePolynomial<D> ret;
    DoMul(self, other, ret);
    return ret;
  }

  static UnivariatePolynomial<D>& MulInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<D>& other) {
    std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    const std::vector<F, base::memory::ReusingAllocator<F>>& r_coefficients =
        other.coefficients_.coefficients_;
    if (self.IsZero() || other.IsOne()) {
      return self;
    } else if (self.IsOne()) {
      l_coefficients = r_coefficients;
      return self;
    } else if (other.IsZero()) {
      l_coefficients = {};
      return self;
    }

    DoMul(self, other, self);
    return self;
  }

  static UnivariatePolynomial<D> Mul(const UnivariatePolynomial<D>& self,
                                     const UnivariatePolynomial<S>& other) {
    if (self.IsZero() || other.IsZero()) {
      return UnivariatePolynomial<D>::Zero();
    } else if (self.IsOne()) {
      return other.ToDense();
    } else if (other.IsOne()) {
      return self;
    }

    UnivariatePolynomial<D> ret;
    DoMul(self, other, ret);
    return ret;
  }

  static UnivariatePolynomial<D>& MulInPlace(
      UnivariatePolynomial<D>& self, const UnivariatePolynomial<S>& other) {
    std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    if (self.IsZero() || other.IsOne()) {
      return self;
    } else if (self.IsOne()) {
      return Copy<false>(self, other);
    } else if (other.IsZero()) {
      l_coefficients = {};
      return self;
    }

    DoMul(self, other, self);
    return self;
  }

  static std::optional<UnivariatePolynomial<D>> Div(
      const UnivariatePolynomial<D>& self, const F& scalar) {
    const std::optional<F> scalar_inv = scalar.Inverse();
    if (LIKELY(scalar_inv)) return Mul(self, *scalar_inv);
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  [[nodiscard]] static std::optional<UnivariatePolynomial<D>*> DivInPlace(
      UnivariatePolynomial<D>& self, const F& scalar) {
    const std::optional<F> scalar_inv = scalar.Inverse();
    if (LIKELY(scalar_inv)) return &MulInPlace(self, *scalar_inv);
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  template <typename DOrS>
  constexpr static std::optional<UnivariatePolynomial<D>> Div(
      const UnivariatePolynomial<D>& self,
      const UnivariatePolynomial<DOrS>& other) {
    DivResult<UnivariatePolynomial<D>> result;
    if (LIKELY(Divide(self, other, result))) {
      result.quotient.coefficients_.RemoveHighDegreeZeros();
      return result.quotient;
    }
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  template <typename DOrS>
  [[nodiscard]] constexpr static std::optional<UnivariatePolynomial<D>*>
  DivInPlace(UnivariatePolynomial<D>& self,
             const UnivariatePolynomial<DOrS>& other) {
    DivResult<UnivariatePolynomial<D>> result;
    if (LIKELY(Divide(self, other, result))) {
      self = std::move(result.quotient);
      self.coefficients_.RemoveHighDegreeZeros();
      return &self;
    }
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  template <typename DOrS>
  constexpr static std::optional<UnivariatePolynomial<D>> Mod(
      const UnivariatePolynomial<D>& self,
      const UnivariatePolynomial<DOrS>& other) {
    DivResult<UnivariatePolynomial<D>> result;
    if (LIKELY(Divide(self, other, result))) {
      result.remainder.coefficients_.RemoveHighDegreeZeros();
      return result.remainder;
    }
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted by mod";
    return std::nullopt;
  }

  template <typename DOrS>
  [[nodiscard]] constexpr static std::optional<UnivariatePolynomial<D>*>
  ModInPlace(UnivariatePolynomial<D>& self,
             const UnivariatePolynomial<DOrS>& other) {
    DivResult<UnivariatePolynomial<D>> result;
    if (LIKELY(Divide(self, other, result))) {
      self = std::move(result.remainder);
      self.coefficients_.RemoveHighDegreeZeros();
      return &self;
    }
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted by mod";
    return std::nullopt;
  }

  template <typename DOrS>
  constexpr static std::optional<DivResult<UnivariatePolynomial<D>>> DivMod(
      const UnivariatePolynomial<D>& self,
      const UnivariatePolynomial<DOrS>& other) {
    DivResult<UnivariatePolynomial<D>> result;
    if (LIKELY(Divide(self, other, result))) return result;
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted by divmod";
    return std::nullopt;
  }

  static const UnivariatePolynomial<D>& ToDense(
      const UnivariatePolynomial<D>& self) {
    return self;
  }

  static UnivariatePolynomial<S> ToSparse(const UnivariatePolynomial<D>& self) {
    std::vector<Term> terms;
    size_t size = self.Degree() + 1;
    // TODO(chokobole): What if this dense polynomial is really sparse..?
    terms.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      if (self[i].IsZero()) {
        continue;
      }
      terms.push_back({i, self[i]});
    }
    return UnivariatePolynomial<S>(S(std::move(terms)));
  }

 private:
  template <bool NEGATION>
  static UnivariatePolynomial<D>& Copy(UnivariatePolynomial<D>& self,
                                       const UnivariatePolynomial<S>& other) {
    std::vector<F, base::memory::ReusingAllocator<F>>& l_coefficients =
        self.coefficients_.coefficients_;
    l_coefficients =
        std::vector<F, base::memory::ReusingAllocator<F>>(other.Degree() + 1);

    const std::vector<Term>& r_terms = other.coefficients().terms_;
    OPENMP_PARALLEL_FOR(const Term& r_term : r_terms) {
      if constexpr (NEGATION) {
        l_coefficients[r_term.degree] = -r_term.coefficient;
      } else {
        l_coefficients[r_term.degree] = r_term.coefficient;
      }
    }
    self.coefficients_.RemoveHighDegreeZeros();
    return self;
  }

  static void DoMul(const UnivariatePolynomial<D>& a,
                    const UnivariatePolynomial<D>& b,
                    UnivariatePolynomial<D>& c) {
    size_t a_degree = a.Degree();
    size_t b_degree = b.Degree();

    const std::vector<F, base::memory::ReusingAllocator<F>>& a_coefficients =
        a.coefficients_.coefficients_;
    const std::vector<F, base::memory::ReusingAllocator<F>>& b_coefficients =
        b.coefficients_.coefficients_;
    std::vector<F, base::memory::ReusingAllocator<F>> c_coefficients(
        a_degree + b_degree + 1);
    for (size_t i = 0; i < b_coefficients.size(); ++i) {
      const F& b = b_coefficients[i];
      if (b.IsZero()) {
        continue;
      } else {
        for (size_t j = 0; j < a_coefficients.size(); ++j) {
          c_coefficients[i + j] += a_coefficients[j] * b;
        }
      }
    }

    c.coefficients_.coefficients_ = std::move(c_coefficients);
    c.coefficients_.RemoveHighDegreeZeros();
  }

  static void DoMul(const UnivariatePolynomial<D>& a,
                    const UnivariatePolynomial<S>& b,
                    UnivariatePolynomial<D>& c) {
    size_t a_degree = a.Degree();
    size_t b_degree = b.Degree();

    const std::vector<F, base::memory::ReusingAllocator<F>>& a_coefficients =
        a.coefficients_.coefficients_;
    const std::vector<Term>& b_terms = b.coefficients().terms_;
    std::vector<F, base::memory::ReusingAllocator<F>> c_coefficients(
        a_degree + b_degree + 1);
    for (size_t i = 0; i < b_terms.size(); ++i) {
      const F& b = b_terms[i].coefficient;
      if (b.IsZero()) {
        continue;
      } else {
        size_t b_degree = b_terms[i].degree;
        for (size_t j = 0; j < a_coefficients.size(); ++j) {
          c_coefficients[b_degree + j] += a_coefficients[j] * b;
        }
      }
    }

    c.coefficients_.coefficients_ = std::move(c_coefficients);
    c.coefficients_.RemoveHighDegreeZeros();
  }

  template <typename DOrS>
  constexpr static bool Divide(const UnivariatePolynomial<D>& self,
                               const UnivariatePolynomial<DOrS>& other,
                               DivResult<UnivariatePolynomial<D>>& output) {
    if (UNLIKELY(other.IsZero())) {
      LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
      return false;
    }
    if (self.IsZero()) {
      output = DivResult<UnivariatePolynomial<D>>{
          UnivariatePolynomial<D>::Zero(), other.ToDense()};
      return true;
    } else if (self.Degree() < other.Degree()) {
      output = DivResult<UnivariatePolynomial<D>>{
          UnivariatePolynomial<D>::Zero(), self.ToDense()};
      return true;
    }
    std::vector<F, base::memory::ReusingAllocator<F>> quotient(
        self.Degree() - other.Degree() + 1);
    UnivariatePolynomial<D> remainder = self.ToDense();
    std::vector<F, base::memory::ReusingAllocator<F>>& r_coefficients =
        remainder.coefficients_.coefficients_;
    F divisor_leading_inv = *other.GetLeadingCoefficient().Inverse();

    while (!remainder.IsZero() && remainder.Degree() >= other.Degree()) {
      F q_coeff =
          remainder.coefficients_.coefficients_.back() * divisor_leading_inv;
      size_t degree = remainder.Degree() - other.Degree();
      quotient[degree] = q_coeff;

      if constexpr (std::is_same_v<DOrS, D>) {
        const std::vector<F, base::memory::ReusingAllocator<F>>& d_terms =
            other.coefficients_.coefficients_;
        for (size_t i = 0; i < d_terms.size(); ++i) {
          r_coefficients[degree + i] -= q_coeff * d_terms[i];
        }
      } else {
        const std::vector<Term>& d_terms = other.coefficients().terms_;
        for (const Term& d_term : d_terms) {
          r_coefficients[degree + d_term.degree] -=
              q_coeff * d_term.coefficient;
        }
      }
      remainder.coefficients_.RemoveHighDegreeZeros();
    }
    D d(std::move(quotient));
    d.RemoveHighDegreeZeros();
    output = DivResult<UnivariatePolynomial<D>>{
        UnivariatePolynomial<D>(std::move(d)), std::move(remainder)};
    return true;
  }
};

template <typename F, size_t MaxDegree>
class UnivariatePolynomialOp<UnivariateSparseCoefficients<F, MaxDegree>> {
 public:
  using D = UnivariateDenseCoefficients<F, MaxDegree>;
  using S = UnivariateSparseCoefficients<F, MaxDegree>;
  using Term = typename S::Term;

  static UnivariatePolynomial<D> Add(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<D>& other) {
    return other + self;
  }

  static UnivariatePolynomial<S> Add(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return other;
    } else if (other.IsZero()) {
      return self;
    }

    UnivariatePolynomial<S> ret;
    DoAdd<false>(self, other, ret);
    return ret;
  }

  static UnivariatePolynomial<S>& AddInPlace(
      UnivariatePolynomial<S>& self, const UnivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return self = other;
    } else if (other.IsZero()) {
      return self;
    }

    DoAdd<false>(self, other, self);
    return self;
  }

  static UnivariatePolynomial<D> Sub(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<D>& other) {
    return -other + self;
  }

  static UnivariatePolynomial<S> Sub(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return -other;
    } else if (other.IsZero()) {
      return self;
    }

    UnivariatePolynomial<S> ret;
    DoAdd<true>(self, other, ret);
    return ret;
  }

  static UnivariatePolynomial<S>& SubInPlace(
      UnivariatePolynomial<S>& self, const UnivariatePolynomial<S>& other) {
    if (self.IsZero()) {
      return self = -other;
    } else if (other.IsZero()) {
      return self;
    }

    DoAdd<true>(self, other, self);
    return self;
  }

  static UnivariatePolynomial<S> Negate(const UnivariatePolynomial<S>& self) {
    if (self.IsZero()) {
      return self;
    }
    const std::vector<Term>& l_terms = self.coefficients_.terms_;
    std::vector<Term> o_terms(l_terms.size());
    for (size_t i = 0; i < o_terms.size(); ++i) {
      o_terms[i] = -l_terms[i];
    }
    return UnivariatePolynomial<S>(S(std::move(o_terms)));
  }

  static UnivariatePolynomial<S>& NegateInPlace(UnivariatePolynomial<S>& self) {
    if (self.IsZero()) {
      return self;
    }
    std::vector<Term>& terms = self.coefficients_.terms_;
    for (Term& term : terms) {
      term.coefficient.NegateInPlace();
    }
    return self;
  }

  static UnivariatePolynomial<D> Mul(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<D>& other) {
    return other * self;
  }

  static UnivariatePolynomial<S> Mul(const UnivariatePolynomial<S>& self,
                                     const F& scalar) {
    if (self.IsZero() || scalar.IsZero()) {
      return UnivariatePolynomial<S>::Zero();
    }
    const std::vector<Term>& l_terms = self.coefficients_.terms_;
    std::vector<Term> o_terms(l_terms.size());
    for (size_t i = 0; i < l_terms.size(); ++i) {
      o_terms[i] = l_terms[i] * scalar;
    }
    return UnivariatePolynomial<S>(S(std::move(o_terms)));
  }

  static UnivariatePolynomial<S>& MulInPlace(UnivariatePolynomial<S>& self,
                                             const F& scalar) {
    std::vector<Term>& terms = self.coefficients_.terms_;
    if (self.IsZero() || scalar.IsOne()) {
      return self;
    } else if (scalar.IsZero()) {
      terms.clear();
      return self;
    }
    for (Term& term : terms) {
      term.coefficient *= scalar;
    }
    return self;
  }

  static UnivariatePolynomial<S> Mul(const UnivariatePolynomial<S>& self,
                                     const UnivariatePolynomial<S>& other) {
    if (self.IsZero() || other.IsZero()) {
      return UnivariatePolynomial<S>::Zero();
    } else if (self.IsOne()) {
      return other;
    } else if (other.IsOne()) {
      return self;
    }

    UnivariatePolynomial<S> ret;
    DoMul(self, other, ret);
    return ret;
  }

  static UnivariatePolynomial<S>& MulInPlace(
      UnivariatePolynomial<S>& self, const UnivariatePolynomial<S>& other) {
    if (self.IsZero() || other.IsOne()) {
      return self;
    } else if (self.IsOne()) {
      return self = other;
    } else if (other.IsZero()) {
      self.coefficients_.terms_.clear();
      return self;
    }

    DoMul(self, other, self);
    return self;
  }

  static std::optional<UnivariatePolynomial<S>> Div(
      const UnivariatePolynomial<S>& self, const F& scalar) {
    const std::optional<F> scalar_inv = scalar.Inverse();
    if (LIKELY(scalar_inv)) return Mul(self, *scalar_inv);
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  [[nodiscard]] static std::optional<UnivariatePolynomial<S>*> DivInPlace(
      UnivariatePolynomial<S>& self, const F& scalar) {
    const std::optional<F> scalar_inv = scalar.Inverse();
    if (LIKELY(scalar_inv)) return &MulInPlace(self, *scalar_inv);
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  template <typename DOrS>
  constexpr static std::optional<UnivariatePolynomial<D>> Div(
      const UnivariatePolynomial<S>& self,
      const UnivariatePolynomial<DOrS>& other) {
    if (self.IsZero()) {
      return UnivariatePolynomial<D>::Zero();
    }
    return self.ToDense() / other;
  }

  template <typename DOrS>
  constexpr static std::optional<UnivariatePolynomial<D>> Mod(
      const UnivariatePolynomial<S>& self,
      const UnivariatePolynomial<DOrS>& other) {
    if (self.IsZero()) {
      return other.ToDense();
    }
    return self.ToDense() % other;
  }

  template <typename DOrS>
  constexpr static std::optional<DivResult<UnivariatePolynomial<D>>> DivMod(
      const UnivariatePolynomial<S>& self,
      const UnivariatePolynomial<DOrS>& other) {
    if (self.IsZero()) {
      return DivResult<UnivariatePolynomial<D>>{UnivariatePolynomial<D>::Zero(),
                                                other.ToDense()};
    }
    return self.ToDense().DivMod(other);
  }

  static UnivariatePolynomial<D> ToDense(const UnivariatePolynomial<S>& self) {
    if (self.IsZero()) {
      return UnivariatePolynomial<D>::Zero();
    }
    size_t size = self.Degree() + 1;
    std::vector<F, base::memory::ReusingAllocator<F>> coefficients(size);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; ++i) {
      coefficients[i] = self[i];
    }
    return UnivariatePolynomial<D>(D(std::move(coefficients)));
  }

  static const UnivariatePolynomial<S>& ToSparse(
      const UnivariatePolynomial<S>& self) {
    return self;
  }

 private:
  template <bool NEGATION>
  static void DoAdd(const UnivariatePolynomial<S>& a,
                    const UnivariatePolynomial<S>& b,
                    UnivariatePolynomial<S>& c) {
    const std::vector<Term>& a_terms = a.coefficients_.terms_;
    const std::vector<Term>& b_terms = b.coefficients_.terms_;
    std::vector<Term> c_terms;
    c_terms.reserve(std::max(a_terms.size(), b_terms.size()));

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
      if (a_it->degree < b_it->degree) {
        c_terms.push_back(*a_it);
        ++a_it;
      } else if (a_it->degree > b_it->degree) {
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
          c_terms.push_back({a_it->degree, std::move(coeff)});
        }
        ++a_it;
        ++b_it;
      }
    }

    c.coefficients_ = S(std::move(c_terms));
  }

  static void DoMul(const UnivariatePolynomial<S>& a,
                    const UnivariatePolynomial<S>& b,
                    UnivariatePolynomial<S>& c) {
    const std::vector<Term>& a_terms = a.coefficients_.terms_;
    const std::vector<Term>& b_terms = b.coefficients_.terms_;
    std::vector<Term> c_terms;
    for (const Term& a_term : a_terms) {
      for (const Term& b_term : b_terms) {
        F f = a_term.coefficient * b_term.coefficient;
        if (f.IsZero()) continue;
        size_t degree = a_term.degree + b_term.degree;
        auto it = base::ranges::find_if(c_terms, [degree](const Term& term) {
          return term.degree == degree;
        });
        if (it != c_terms.end()) {
          it->coefficient += f;
          if (it->coefficient.IsZero()) {
            c_terms.erase(it);
          }
        } else {
          c_terms.push_back({degree, std::move(f)});
        }
      }
    }
    pdqsort(c_terms.begin(), c_terms.end());
    c.coefficients_ = S(std::move(c_terms));
  }
};

}  // namespace internal
}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_OPS_H_
