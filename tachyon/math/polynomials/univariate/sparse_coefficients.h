#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_SPARSE_COEFFICIENTS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_SPARSE_COEFFICIENTS_H_

#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <utility>
#include <vector>

#include "absl/numeric/internal/bits.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial_ops_forward.h"

namespace tachyon::math {

template <typename F, size_t MaxDegree>
class DenseCoefficients;

template <typename F, size_t MaxDegree>
class SparseCoefficients {
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

  constexpr SparseCoefficients() = default;
  constexpr explicit SparseCoefficients(const std::vector<Term>& terms)
      : terms_(terms) {
    CHECK_LE(Degree(), kMaxDegree);
    DCHECK(base::ranges::is_sorted(terms_.begin(), terms_.end()));
    RemoveHighDegreeZeros();
  }
  constexpr explicit SparseCoefficients(std::vector<Term>&& terms)
      : terms_(std::move(terms)) {
    CHECK_LE(Degree(), kMaxDegree);
    DCHECK(base::ranges::is_sorted(terms_.begin(), terms_.end()));
    RemoveHighDegreeZeros();
  }

  constexpr static SparseCoefficients Zero() { return SparseCoefficients(); }

  constexpr static SparseCoefficients One() {
    return SparseCoefficients({{0, Field::One()}});
  }

  constexpr static SparseCoefficients Random(size_t degree) {
    // TODO(chokobole): Better idea?
    std::vector<Term> terms;
    for (size_t i = 0; i < degree + 1; ++i) {
      F f = F::Random();
      if (f.IsZero()) continue;
      terms.push_back({i, std::move(f)});
    }
    return SparseCoefficients(std::move(terms));
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

  constexpr Field* Get(size_t i) {
    return const_cast<Field*>(std::as_const(*this).Get(i));
  }

  constexpr const Field* Get(size_t i) const {
    auto it = std::lower_bound(
        terms_.begin(), terms_.end(), i,
        [](const Term& term, size_t degree) { return term.degree < degree; });
    if (it == terms_.end()) return nullptr;
    if (it->degree != i) return nullptr;
    return &it->coefficient;
  }

  constexpr const Field* GetLeadingCoefficient() const {
    if (IsZero()) return nullptr;
    return &terms_.back().coefficient;
  }

  constexpr bool IsZero() const { return terms_.empty(); }

  constexpr bool IsOne() const {
    return terms_.size() == 1 && terms_[0].coefficient.IsOne();
  }

  constexpr size_t Degree() const {
    if (IsZero()) return 0;
    return terms_.back().degree;
  }

  constexpr Field Evaluate(const Field& point) const {
    if (IsZero()) return Field::Zero();

    static_assert(sizeof(size_t) == sizeof(uint64_t));
    size_t num_powers = absl::numeric_internal::CountLeadingZeroes64(0) -
                        absl::numeric_internal::CountLeadingZeroes64(Degree());
    std::vector<Field> powers_of_2;
    powers_of_2.reserve(num_powers);

    Field p = point;
    powers_of_2.push_back(p);
    for (size_t i = 1; i < num_powers; ++i) {
      p.SquareInPlace();
      powers_of_2.push_back(p);
    }

    Field sum = Field::Zero();
    for (const Term& term : terms_) {
      sum += Field::PowWithTable(absl::MakeConstSpan(powers_of_2),
                                 Field(term.degree).ToBigInt()) *
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
      DenseCoefficients<F, MaxDegree>>;
  friend class internal::UnivariatePolynomialOp<
      SparseCoefficients<F, MaxDegree>>;

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
                         const SparseCoefficients<F, MaxDegree>& p) {
  return os << p.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_SPARSE_COEFFICIENTS_H_
