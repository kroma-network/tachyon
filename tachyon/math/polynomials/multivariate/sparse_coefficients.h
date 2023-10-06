#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_SPARSE_COEFFICIENTS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_SPARSE_COEFFICIENTS_H_

#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/internal/bits.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/cxx20_erase_vector.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/random.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/polynomials/multivariate/multivariate_polynomial_ops_forward.h"

namespace tachyon::math {
template <typename F, size_t MaxDegree>
class SparseCoefficients {
 public:
  constexpr static const size_t kMaxDegree = MaxDegree;

  using Field = F;

  struct Element {
    size_t variable = 0;
    size_t exponent = 0;

    bool operator<(const Element& other) const {
      if (exponent != 0 && other.exponent != 0) {
        if (variable != other.variable) {
          return variable < other.variable;
        }
        // At this point, variables are equal, so compare the exponents.
        return exponent < other.exponent;
      }
      // If either exponent is zero, compare the exponents directly.
      // When both exponents are zero, the order based on the variable does not
      // matter since constant terms are compacted. See the Compact() method.
      return exponent < other.exponent;
    }
    bool operator==(const Element& other) const {
      return variable == other.variable && exponent == other.exponent;
    }
    bool operator!=(const Element& other) const {
      return variable != other.variable || exponent != other.exponent;
    }
  };

  struct Literal {
    std::vector<Element> elements;

    Literal() = default;

    // NOLINTNEXTLINE(runtime/explicit)
    Literal(const std::vector<Element>& elems) : elements(elems) {
      DCHECK(base::ranges::is_sorted(elements.begin(), elements.end()));
    }

    // NOLINTNEXTLINE(runtime/explicit)
    Literal(std::vector<Element>&& elems) : elements(std::move(elems)) {
      DCHECK(base::ranges::is_sorted(elements.begin(), elements.end()));
    }

    bool operator<(const Literal& other) const {
      if (elements.back() == other.elements.back()) {
        return elements.size() < other.elements.size();
      } else {
        return elements.back() < other.elements.back();
      }
    }
    bool operator==(const Literal& other) const {
      return (Degree() == 0 && other.Degree() == 0) ||
             elements == other.elements;
    }
    bool operator!=(const Literal& other) const {
      return (Degree() != 0 || other.Degree() != 0) &&
             elements != other.elements;
    }

    size_t Degree() const {
      if (elements.empty()) return 0;
      return std::accumulate(elements.begin(), elements.end(), 0,
                             [](const size_t acc, const Element& elem) {
                               return acc + elem.exponent;
                             });
    }

    Field Evaluate(const std::vector<Field>& points) const {
      return std::accumulate(
          elements.begin(), elements.end(), Field::One(),
          [&points](const Field& acc, const Element& elem) {
            return acc * points[elem.variable].Pow(BigInt<1>(elem.exponent));
          });
    }

    std::string ToString() const {
      std::stringstream ss;
      for (const Element& elem : base::Reversed(elements)) {
        ss << "x_" << elem.variable;
        if (elem.exponent > 1) {
          ss << "^" << elem.exponent;
        }
      }
      return ss.str();
    }
  };

  struct Term {
    Literal literal;
    Field coefficient;

    Term operator-() const { return {literal, -coefficient}; }

    bool operator<(const Term& other) const {
      if (literal != other.literal) {
        return literal < other.literal;
      } else {
        return coefficient < other.coefficient;
      }
    }
    bool operator==(const Term& other) const {
      return literal == other.literal && coefficient == other.coefficient;
    }
    bool operator!=(const Term& other) const {
      return literal != other.literal || coefficient != other.coefficient;
    }

    size_t Degree() const { return literal.Degree(); }

    static Term Constant(Field field) { return {{{{}}}, field}; }
  };

  using Terms = std::vector<Term>;

  constexpr SparseCoefficients() = default;
  constexpr SparseCoefficients(size_t num_vars, const Terms& terms)
      : num_vars_(num_vars), terms_(terms) {
    CHECK_LE(Degree(), kMaxDegree);
    DCHECK(base::ranges::is_sorted(terms_.begin(), terms_.end()));
  }
  constexpr SparseCoefficients(size_t num_vars, Terms&& terms)
      : num_vars_(num_vars), terms_(std::move(terms)) {
    CHECK_LE(Degree(), kMaxDegree);
    DCHECK(base::ranges::is_sorted(terms_.begin(), terms_.end()));
  }

  constexpr static SparseCoefficients Zero() { return SparseCoefficients(); }

  constexpr static SparseCoefficients One() {
    return SparseCoefficients(1, {{{{{0, 0}}}, Field::One()}});
  }

  static SparseCoefficients Random(size_t arity, size_t exponent,
                                   size_t min_term = 1, size_t max_term = 999) {
    Terms terms;
    size_t num_terms = base::Uniform(min_term, max_term);
    terms.push_back(Term::Constant(Field::Random()));
    for (size_t i = 1; i < num_terms; ++i) {
      for (size_t j = 0; j < arity; ++j) {
        if (base::Bernoulli(0.5) > 0.5) {
          terms.insert(
              terms.begin(),
              {{{{j, base::Uniform(static_cast<size_t>(0), exponent)}}},
               Field::Random()});
        }
      }
    }
    base::ranges::sort(terms.begin(), terms.end());
    return SparseCoefficients(arity, std::move(terms));
  }

  constexpr bool IsConstant() const {
    return terms_.size() == 1 && terms_[0].Degree() == 0;
  }

  constexpr bool operator==(const SparseCoefficients& other) const {
    if (IsZero()) {
      return other.IsZero();
    }
    if (other.IsZero()) {
      return false;
    }
    return terms_ == other.terms_;
  }

  constexpr bool operator!=(const SparseCoefficients& other) const {
    return !operator==(other);
  }

  constexpr Field* Get(const Literal& literal) {
    return const_cast<Field*>(std::as_const(*this).Get(literal));
  }

  constexpr const Field* Get(const Literal& literal) const {
    auto it = std::lower_bound(terms_.begin(), terms_.end(), literal,
                               [](const Term& term, const Literal& literal) {
                                 return term.literal < literal;
                               });
    if (it != terms_.end() && it->literal == literal) {
      return &it->coefficient;
    }
    return nullptr;
  }

  constexpr const Field* GetLeadingCoefficient() const {
    if (IsZero()) return nullptr;
    return &terms_.begin().coefficient;
  }

  constexpr bool IsZero() const { return terms_.empty(); }

  constexpr bool IsOne() const {
    return terms_.size() == 1 && terms_[0].Degree() == 0 &&
           terms_[0].coefficient.IsOne();
  }

  constexpr size_t Degree() const {
    if (IsConstant()) return 0;
    return terms_.back().Degree();
  }

  constexpr Field Evaluate(const std::vector<Field>& points) const {
    CHECK_LE(points.size(), num_vars_) << "Invalid evaluation domain";
    if (IsZero()) {
      return Field::Zero();
    }
    return std::accumulate(terms_.begin(), terms_.end(), Field::Zero(),
                           [&points](const Field& acc, const Term& term) {
                             return acc + term.coefficient *
                                              term.literal.Evaluate(points);
                           });
  }

  std::string ToString() const {
    if (IsZero()) return base::EmptyString();
    std::stringstream ss;
    bool has_term = false;
    for (const Term& term : base::Reversed(terms_)) {
      if (has_term) ss << " + ";
      has_term = true;
      ss << term.coefficient
                .ToString();  // No need to consider the case where
                              // the coefficient is 0 since we
                              // compact the polynomial on initialization.
      if (term.literal.Degree() == 0) {
        // do nothing
      } else {
        ss << " * " << term.literal.ToString();
      }
    }
    return ss.str();
  }

  void Compact() {
    auto write_itr = terms_.begin();
    for (auto read_itr = terms_.begin() + 1; read_itr != terms_.end();
         ++read_itr) {
      if (write_itr->literal == read_itr->literal) {
        write_itr->coefficient += read_itr->coefficient;
      } else {
        ++write_itr;
        *write_itr = *read_itr;
      }
    }
    terms_.erase(write_itr + 1, terms_.end());
    // Compact the constant term.
    Field constant = Field::Zero();
    while (terms_.size() > 0 && terms_.front().Degree() == 0) {
      constant += terms_.front().coefficient;
      terms_.erase(terms_.begin());
    }
    terms_.insert(terms_.begin(), Term::Constant(constant));
    base::EraseIf(terms_,
                  [](const Term& term) { return term.coefficient.IsZero(); });
  }

 private:
  friend class internal::MultivariatePolynomialOp<
      SparseCoefficients<F, MaxDegree>>;

  size_t num_vars_;
  std::vector<Term> terms_;
};

template <typename F, size_t MaxDegree>
std::ostream& operator<<(std::ostream& os,
                         const SparseCoefficients<F, MaxDegree>& p) {
  return os << p.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_SPARSE_COEFFICIENTS_H_
