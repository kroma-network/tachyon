#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_SPARSE_COEFFICIENTS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_SPARSE_COEFFICIENTS_H_

#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/internal/bits.h"
#include "absl/types/span.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial_ops_forward.h"

namespace tachyon::math {

template <typename F, size_t MaxDegree>
class UnivariateDenseCoefficients;

// SparseCoefficients class provides a representation for polynomials where
// only non-zero coefficients are stored. This is efficient for polynomials
// where most of the degrees have zero coefficients.
template <typename F, size_t MaxDegree>
class UnivariateSparseCoefficients {
 public:
  constexpr static const size_t kMaxDegree = MaxDegree;

  using Field = F;

  struct Term {
    size_t degree;
    F coefficient;

    Term operator-() const { return {degree, -coefficient}; }

    bool operator<(const Term& other) const { return degree < other.degree; }
    bool operator==(const Term& other) const {
      return degree == other.degree && coefficient == other.coefficient;
    }
    bool operator!=(const Term& other) const {
      return degree != other.degree || coefficient != other.coefficient;
    }
  };

  constexpr UnivariateSparseCoefficients() = default;
  constexpr explicit UnivariateSparseCoefficients(
      const std::vector<Term>& terms)
      : terms_(terms) {
    CHECK_LE(Degree(), kMaxDegree);
    DCHECK(base::ranges::is_sorted(terms_.begin(), terms_.end()));
    RemoveHighDegreeZeros();
  }
  constexpr explicit UnivariateSparseCoefficients(std::vector<Term>&& terms)
      : terms_(std::move(terms)) {
    CHECK_LE(Degree(), kMaxDegree);
    DCHECK(base::ranges::is_sorted(terms_.begin(), terms_.end()));
    RemoveHighDegreeZeros();
  }

  constexpr static UnivariateSparseCoefficients Zero() {
    return UnivariateSparseCoefficients();
  }

  constexpr static UnivariateSparseCoefficients One() {
    return UnivariateSparseCoefficients({{0, F::One()}});
  }

  constexpr static UnivariateSparseCoefficients Random(size_t degree) {
    // TODO(chokobole): Better idea?
    std::vector<Term> terms;
    for (size_t i = 0; i < degree + 1; ++i) {
      F f = F::Random();
      if (f.IsZero()) continue;
      terms.push_back({i, std::move(f)});
    }
    return UnivariateSparseCoefficients(std::move(terms));
  }

  constexpr bool operator==(const UnivariateSparseCoefficients& other) const {
    return terms_ == other.terms_;
  }

  constexpr bool operator!=(const UnivariateSparseCoefficients& other) const {
    return !operator==(other);
  }

  constexpr F* Get(size_t i) {
    return const_cast<F*>(std::as_const(*this).Get(i));
  }

  constexpr const F* Get(size_t i) const {
    auto it = std::lower_bound(
        terms_.begin(), terms_.end(), i,
        [](const Term& term, size_t degree) { return term.degree < degree; });
    if (it == terms_.end()) return nullptr;
    if (it->degree != i) return nullptr;
    return &it->coefficient;
  }

  constexpr const F* GetLeadingCoefficient() const {
    if (IsZero()) return nullptr;
    return &terms_.back().coefficient;
  }

  constexpr bool IsZero() const { return terms_.empty(); }

  constexpr bool IsOne() const {
    return terms_.size() == 1 && terms_[0].degree == 0 &&
           terms_[0].coefficient.IsOne();
  }

  constexpr size_t Degree() const {
    if (IsZero()) return 0;
    return terms_.back().degree;
  }

  constexpr F Evaluate(const F& point) const {
    if (IsZero()) return F::Zero();

    static_assert(sizeof(size_t) == sizeof(uint64_t));
    size_t num_powers = absl::numeric_internal::CountLeadingZeroes64(0) -
                        absl::numeric_internal::CountLeadingZeroes64(Degree());
    std::vector<F> powers_of_2;
    powers_of_2.reserve(num_powers);

    F p = point;
    powers_of_2.push_back(p);
    for (size_t i = 1; i < num_powers; ++i) {
      p.SquareInPlace();
      powers_of_2.push_back(p);
    }

    F sum = F::Zero();
    for (const Term& term : terms_) {
      sum += F::PowWithTable(absl::MakeConstSpan(powers_of_2),
                             F(term.degree).ToBigInt()) *
             term.coefficient;
    }
    return sum;
  }

  std::string ToString() const {
    if (IsZero()) return base::EmptyString();
    std::stringstream ss;
    bool has_term = false;
    for (const Term& term : base::Reversed(terms_)) {
      if (has_term) ss << " + ";
      has_term = true;
      ss << term.coefficient.ToString();
      if (term.degree == 0) {
        // do nothing
      } else if (term.degree == 1) {
        ss << " * x";
      } else {
        ss << " * x^" << term.degree;
      }
    }
    return ss.str();
  }

 private:
  friend class internal::UnivariatePolynomialOp<
      UnivariateDenseCoefficients<F, MaxDegree>>;
  friend class internal::UnivariatePolynomialOp<
      UnivariateSparseCoefficients<F, MaxDegree>>;

  void RemoveHighDegreeZeros() {  // Fix to RemoveZeros
    while (!IsZero()) {
      if (terms_.back().coefficient.IsZero()) {
        terms_.pop_back();
      } else {
        break;
      }
    }
  }

  std::vector<Term> terms_;
};

template <typename F, size_t MaxDegree>
std::ostream& operator<<(std::ostream& os,
                         const UnivariateSparseCoefficients<F, MaxDegree>& p) {
  return os << p.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_SPARSE_COEFFICIENTS_H_
