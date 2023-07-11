#ifndef TACHYON_MATH_POLYNOMIALS_SPARSE_COEFFICIENTS_H_
#define TACHYON_MATH_POLYNOMIALS_SPARSE_COEFFICIENTS_H_

#include <stddef.h>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "absl/numeric/internal/bits.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/univariate_polynomial_ops_forward.h"

namespace tachyon {
namespace math {

template <typename F, size_t MAX_DEGREE>
class DenseCoefficients;

template <typename F, size_t _MAX_DEGREE>
class SparseCoefficients {
 public:
  constexpr static const size_t MAX_DEGREE = _MAX_DEGREE;

  using Field = F;

  struct Element {
    size_t degree;
    F coefficient;

    Element operator-() const { return {degree, -coefficient}; }

    bool operator<(const Element& other) const { return degree < other.degree; }
    bool operator==(const Element& other) const {
      return degree == other.degree && coefficient == other.coefficient;
    }
    bool operator!=(const Element& other) const {
      return degree != other.degree || coefficient != other.coefficient;
    }
  };

  constexpr SparseCoefficients() = default;
  constexpr explicit SparseCoefficients(const std::vector<Element>& elements)
      : elements_(elements) {
    CHECK_LE(Degree(), MAX_DEGREE);
    DCHECK(base::ranges::is_sorted(elements_.begin(), elements_.end()));
    RemoveHighDegreeZeros();
  }
  constexpr explicit SparseCoefficients(std::vector<Element>&& elements)
      : elements_(std::move(elements)) {
    CHECK_LE(Degree(), MAX_DEGREE);
    DCHECK(base::ranges::is_sorted(elements_.begin(), elements_.end()));
    RemoveHighDegreeZeros();
  }

  constexpr static SparseCoefficients Zero() { return SparseCoefficients(); }

  constexpr static SparseCoefficients One() {
    return SparseCoefficients({{0, Field::One()}});
  }

  constexpr static SparseCoefficients Random(size_t degree) {
    // TODO(chokobole): Better idea?
    std::vector<Element> elements;
    for (size_t i = 0; i < degree + 1; ++i) {
      F f = F::Random();
      if (f.IsZero()) continue;
      elements.push_back({i, std::move(f)});
    }
    return SparseCoefficients(std::move(elements));
  }

  constexpr bool operator==(const SparseCoefficients& other) const {
    if (IsZero()) {
      return other.IsZero();
    }
    if (other.IsZero()) {
      return false;
    }
    return elements_ == other.elements_;
  }

  constexpr bool operator!=(const SparseCoefficients& other) const {
    return !operator==(other);
  }

  constexpr Field* Get(size_t i) {
    return const_cast<Field*>(std::as_const(*this).Get(i));
  }

  constexpr const Field* Get(size_t i) const {
    auto it = std::lower_bound(elements_.begin(), elements_.end(), i,
                               [](const Element& element, size_t degree) {
                                 return element.degree < degree;
                               });
    if (it == elements_.end()) return nullptr;
    if (it->degree != i) return nullptr;
    return &it->coefficient;
  }

  constexpr const Field* GetLeadingCoefficient() const {
    if (elements_.empty()) return nullptr;
    return &elements_.back().coefficient;
  }

  constexpr bool IsZero() const {
    return elements_.empty() ||
           (elements_.size() == 1 && elements_[0].coefficient.IsZero());
  }

  constexpr bool IsOne() const {
    return elements_.size() == 1 && elements_[0].coefficient.IsOne();
  }

  constexpr size_t Degree() const {
    if (elements_.empty()) return 0;
    return elements_.back().degree;
  }

  constexpr Field Evaluate(const Field& point) const {
    if (elements_.empty()) return Field::Zero();

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
    for (const Element& element : elements_) {
      sum += Field::PowWithTable(absl::MakeConstSpan(powers_of_2),
                                 mpz_class(element.degree)) *
             element.coefficient;
    }
    return sum;
  }

  std::string ToString() const {
    if (elements_.empty()) return base::EmptyString();
    size_t len = elements_.size() - 1;
    std::stringstream ss;
    bool has_elem = false;
    while (len >= 0) {
      size_t i = len--;
      const Element& elem = elements_[i];
      if (has_elem) ss << " + ";
      has_elem = true;
      ss << elem.coefficient.ToString();
      if (elem.degree == 0) {
        // do nothing
      } else if (elem.degree == 1) {
        ss << " * x";
      } else {
        ss << " * x^" << elem.degree;
      }
      if (i == 0) break;
    }
    return ss.str();
  }

 private:
  friend class internal::UnivariatePolynomialOp<
      DenseCoefficients<F, MAX_DEGREE>>;
  friend class internal::UnivariatePolynomialOp<
      SparseCoefficients<F, MAX_DEGREE>>;

  void RemoveHighDegreeZeros() {
    while (!elements_.empty()) {
      if (elements_.back().coefficient.IsZero()) {
        elements_.pop_back();
      } else {
        break;
      }
    }
  }

  std::vector<Element> elements_;
};

template <typename F, size_t MAX_DEGREE>
std::ostream& operator<<(std::ostream& os,
                         const SparseCoefficients<F, MAX_DEGREE>& p) {
  return os << p.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_SPARSE_COEFFICIENTS_H_
