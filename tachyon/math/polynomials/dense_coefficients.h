#ifndef TACHYON_MATH_POLYNOMIALS_DENSE_COEFFICIENTS_H_
#define TACHYON_MATH_POLYNOMIALS_DENSE_COEFFICIENTS_H_

#include <stddef.h>

#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/univariate_polynomial_ops_forward.h"

namespace tachyon {
namespace math {

template <typename F, size_t MAX_DEGREE>
class SparseCoefficients;

template <typename F, size_t _MAX_DEGREE>
class DenseCoefficients {
 public:
  constexpr static const size_t MAX_DEGREE = _MAX_DEGREE;

  using Field = F;

  constexpr DenseCoefficients() = default;
  constexpr explicit DenseCoefficients(const std::vector<F>& coefficients)
      : coefficients_(coefficients) {
    CHECK_LE(Degree(), MAX_DEGREE);
  }
  constexpr explicit DenseCoefficients(std::vector<F>&& coefficients)
      : coefficients_(std::move(coefficients)) {
    CHECK_LE(Degree(), MAX_DEGREE);
  }

  constexpr static DenseCoefficients Random(size_t degree) {
    return DenseCoefficients(
        base::CreateVector(degree + 1, []() { return F::Random(); }));
  }

  constexpr bool operator==(const DenseCoefficients& other) const {
    return coefficients_ == other.coefficients_;
  }

  constexpr bool operator!=(const DenseCoefficients& other) const {
    return coefficients_ != other.coefficients_;
  }

  constexpr Field* Get(size_t i) {
    return const_cast<Field*>(std::as_const(*this).Get(i));
  }

  constexpr const Field* Get(size_t i) const {
    if (i < coefficients_.size()) {
      return &coefficients_[i];
    }
    return nullptr;
  }

  constexpr bool IsEmpty() const { return coefficients_.empty(); }

  constexpr size_t Degree() const {
    if (coefficients_.empty()) return 0;
    return coefficients_.size() - 1;
  }

  constexpr Field Evaluate(const Field& point) const {
    if (coefficients_.empty()) return Field::Zero();
    if (point.IsZero()) return coefficients_[0];
    return DoEvaluate(point);
  }

  std::string ToString() const {
    if (coefficients_.empty()) return base::EmptyString();
    size_t len = coefficients_.size() - 1;
    std::stringstream ss;
    bool has_coeff = false;
    while (len >= 0) {
      size_t i = len--;
      const Field& coeff = coefficients_[i];
      if (!coeff.IsZero()) {
        if (has_coeff) ss << " + ";
        has_coeff = true;
        ss << coeff.ToString();
        if (i == 0) {
          // do nothing
        } else if (i == 1) {
          ss << " * x";
        } else {
          ss << " * x^" << i;
        }
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

  constexpr Field DoEvaluate(const Field& point) const {
    return HornerEvaluate(point);
  }

  constexpr Field HornerEvaluate(const Field& point) const {
    return std::accumulate(coefficients_.rbegin(), coefficients_.rend(),
                           Field::Zero(),
                           [&point](Field result, const Field& coeff) {
                             return result * point + coeff;
                           });
  }

  std::vector<Field> coefficients_;
};

template <typename F, size_t MAX_DEGREE>
std::ostream& operator<<(std::ostream& os,
                         const DenseCoefficients<F, MAX_DEGREE>& p) {
  return os << p.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_DENSE_COEFFICIENTS_H_
