// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_DENSE_COEFFICIENTS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_DENSE_COEFFICIENTS_H_

#include <stddef.h>

#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/univariate/support_poly_operators.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_forwards.h"

namespace tachyon {
namespace math {

template <typename Coefficients>
class UnivariatePolynomial;

template <typename F, size_t MaxDegree>
class UnivariateSparseCoefficients;

// DenseCoefficients class provides a representation for polynomials where
// coefficients are stored contiguously for each degree. This is efficient for
// polynomials where most of the degrees have non-zero coefficients.
template <typename F, size_t MaxDegree>
class UnivariateDenseCoefficients {
 public:
  constexpr static size_t kMaxDegree = MaxDegree;

  constexpr static F kZero = F::Zero();

  using Field = F;
  using Point = F;

  constexpr UnivariateDenseCoefficients() = default;
  constexpr explicit UnivariateDenseCoefficients(
      const std::vector<F>& coefficients)
      : coefficients_(coefficients) {
    CHECK_LE(Degree(), kMaxDegree);
    RemoveHighDegreeZeros();
  }
  constexpr explicit UnivariateDenseCoefficients(std::vector<F>&& coefficients)
      : coefficients_(std::move(coefficients)) {
    CHECK_LE(Degree(), kMaxDegree);
    RemoveHighDegreeZeros();
  }

  constexpr static UnivariateDenseCoefficients Zero() {
    return UnivariateDenseCoefficients();
  }

  constexpr static UnivariateDenseCoefficients One() {
    return UnivariateDenseCoefficients({F::One()});
  }

  constexpr static UnivariateDenseCoefficients Random(size_t degree) {
    return UnivariateDenseCoefficients(
        base::CreateVector(degree + 1, []() { return F::Random(); }));
  }

  // Return dense coefficients according to the given |roots|.
  // This is taken and modified from
  // https://github.com/Plonky3/Plonky3/blob/b21d54f13fd7949a2661c9478b91c01bc3abccbe/field/src/helpers.rs#L81-L92.
  template <typename Container>
  constexpr static UnivariateDenseCoefficients FromRoots(
      const Container& roots) {
    // clang-format off
    // For (X - x₀)(X - x₁)(X - x₂)(X - x₃), what this function does looks as follows:
    //
    //       |     c[0] |             c[1] |             c[2] |             c[3] |             c[4] |
    // ------|----------|------------------|------------------|------------------| -----------------|
    // init  |        1 |               0  |               0  |               0  |               0  |
    // i = 0 |      -x₀ | c[0] - x₀ * c[1] | c[1] - x₀ * c[2] | c[2] - x₀ * c[3] | c[3] - x₀ * c[4] |
    // i = 1 |     x₀x₁ | c[0] - x₁ * c[1] | c[1] - x₁ * c[2] | c[2] - x₁ * c[3] | c[3] - x₁ * c[4] |
    // i = 2 |  -x₀x₁x₂ | c[0] - x₂ * c[1] | c[1] - x₂ * c[2] | c[2] - x₂ * c[3] | c[3] - x₂ * c[4] |
    // i = 3 | x₀x₁x₂x₃ | c[0] - x₃ * c[1] | c[1] - x₃ * c[2] | c[2] - x₃ * c[3] | c[3] - x₃ * c[4] |

    // Then the values are changed as follows:
    //
    //       |     c[0] |                                 c[1] |                                    c[2] |                 c[3] | c[4] |
    // ------|----------|--------------------------------------|-----------------------------------------|----------------------|------|
    // init  |        1 |                                    0 |                                       0 |                    0 |    0 |
    // i = 0 |      -x₀ |                                    1 |                                       0 |                    0 |    0 |
    // i = 1 |     x₀x₁ |                           -(x₀ + x₁) |                                       1 |                    0 |    0 |
    // i = 2 |  -x₀x₁x₂ |                   x₀x₁ + x₀x₂ + x₁x₂ |                          -(x₀ + x₁ +x₂) |                    1 |    0 |
    // i = 3 | x₀x₁x₂x₃ | -(x₀x₁x₂ + x₀x₁x₃ + x₀x₂x₃ + x₁x₂x₃) | x₀x₁ + x₀x₂ + x₀x₃ + x₁x₂ + x₁x₃ + x₂x₃ | -(x₀ + x₁ + x₂ + x₃) |    1 |
    // clang-format on

    std::vector<F> coefficients(std::size(roots) + 1);
    coefficients[0] = F::One();
    for (size_t i = 0; i < std::size(roots); ++i) {
      for (size_t j = i + 1; j > 0; --j) {
        coefficients[j] = coefficients[j - 1] - roots[i] * coefficients[j];
      }
      coefficients[0] *= -roots[i];
    }

    UnivariateDenseCoefficients ret;
    ret.coefficients_ = std::move(coefficients);
    return ret;
  }

  constexpr const std::vector<F>& coefficients() const { return coefficients_; }
  constexpr std::vector<F>& coefficients() { return coefficients_; }

  std::vector<F>&& TakeCoefficients() && { return std::move(coefficients_); }

  constexpr bool operator==(const UnivariateDenseCoefficients& other) const {
    return coefficients_ == other.coefficients_;
  }

  constexpr bool operator!=(const UnivariateDenseCoefficients& other) const {
    return !operator==(other);
  }

  constexpr F& at(size_t i) {
    CHECK_LT(i, coefficients_.size());
    return coefficients_[i];
  }
  constexpr const F& at(size_t i) const { return (*this)[i]; }

  constexpr const F& operator[](size_t i) const {
    if (i < coefficients_.size()) {
      return coefficients_[i];
    }
    return kZero;
  }

  constexpr const F& GetLeadingCoefficient() const {
    if (IsZero()) return kZero;
    return coefficients_.back();
  }

  constexpr bool IsZero() const { return coefficients_.empty(); }

  constexpr bool IsOne() const {
    return coefficients_.size() == 1 && coefficients_[0].IsOne();
  }

  constexpr size_t Degree() const {
    if (IsZero()) return 0;
    return coefficients_.size() - 1;
  }

  constexpr size_t NumElements() const { return coefficients_.size(); }

  constexpr F Evaluate(const Point& point) const {
    if (IsZero()) return F::Zero();
    if (point.IsZero()) return coefficients_[0];
    return DoEvaluate(point);
  }

  // Return coefficients where the original coefficients reduce their degree
  // by categorizing coefficients into even and odd degrees,
  // multiplying either set of coefficients by a specified random field |r|,
  // and summing them together.
  template <bool MulRandomWithEvens>
  CONSTEXPR_IF_NOT_OPENMP UnivariateDenseCoefficients
  Fold(const Field& r) const {
    size_t size = coefficients_.size();
    std::vector<F> coefficients((size + 1) >> 1);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; i += 2) {
      if constexpr (MulRandomWithEvens) {
        coefficients[i >> 1] = coefficients_[i] * r;
        coefficients[i >> 1] += coefficients_[i + 1];
      } else {
        coefficients[i >> 1] = coefficients_[i + 1] * r;
        coefficients[i >> 1] += coefficients_[i];
      }
    }
    if (size % 2 != 0) {
      coefficients[size >> 1] = coefficients_[size - 1];
      if constexpr (MulRandomWithEvens) {
        coefficients[size >> 1] *= r;
      }
    }
    return UnivariateDenseCoefficients(std::move(coefficients));
  }

  std::string ToString() const {
    if (IsZero()) return base::EmptyString();
    size_t len = coefficients_.size() - 1;
    std::stringstream ss;
    bool has_coeff = false;
    for (const F& coeff : base::Reversed(coefficients_)) {
      size_t i = len--;
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
    }
    return ss.str();
  }

 private:
  friend class internal::UnivariatePolynomialOp<
      UnivariateDenseCoefficients<F, MaxDegree>>;
  friend class internal::UnivariatePolynomialOp<
      UnivariateSparseCoefficients<F, MaxDegree>>;
  friend class UnivariatePolynomial<UnivariateDenseCoefficients<F, MaxDegree>>;
  friend class UnivariateEvaluationDomain<F, MaxDegree>;
  friend class Radix2EvaluationDomain<F, MaxDegree>;
  friend class MixedRadixEvaluationDomain<F, MaxDegree>;
  friend class base::Copyable<UnivariateDenseCoefficients<F, MaxDegree>>;

  // NOTE(chokobole): This doesn't call |RemoveHighDegreeZeros()| internally.
  // So when the returned instance of |UnivariateDenseCoefficients| is called
  // with |IsZero()|, it returns false. So please use it carefully!
  constexpr static UnivariateDenseCoefficients Zero(size_t degree) {
    UnivariateDenseCoefficients ret;
    ret.coefficients_ = std::vector<F>(degree + 1);
    return ret;
  }

  constexpr F DoEvaluate(const Point& point) const {
    // Horner's method - parallel method
    // run Horner's method on each thread as follows:
    // 1) Split up the coefficients across each thread evenly.
    // 2) Do polynomial evaluation via Horner's method for the thread's
    // coefficients
    // 3) Scale the result point^{thread coefficient start index}
    // Then obtain the final polynomial evaluation by summing each threads
    // result.
    std::vector<F> results = base::ParallelizeMap(
        coefficients_, [&point](absl::Span<const F> chunk, size_t chunk_offset,
                                size_t chunk_size) {
          return HornerEvaluate(chunk, point) *
                 point.Pow(chunk_offset * chunk_size);
        });
    return std::accumulate(results.begin(), results.end(), F::Zero(),
                           std::plus<>());
  }

  constexpr static F HornerEvaluate(absl::Span<const F> coefficients,
                                    const Point& point) {
    return std::accumulate(coefficients.rbegin(), coefficients.rend(),
                           F::Zero(), [&point](F& result, const F& coeff) {
                             result *= point;
                             return result += coeff;
                           });
  }

  void RemoveHighDegreeZeros() {
    while (!IsZero()) {
      if (coefficients_.back().IsZero()) {
        coefficients_.pop_back();
      } else {
        break;
      }
    }
  }

  std::vector<F> coefficients_;
};

template <typename H, typename F, size_t MaxDegree>
H AbslHashValue(H h,
                const UnivariateDenseCoefficients<F, MaxDegree>& coefficients) {
  // NOTE(chokobole): We shouldn't hash only with a non-zero term.
  // See https://abseil.io/docs/cpp/guides/hash#the-abslhashvalue-overload
  size_t degree = 0;
  for (const F& coefficient : coefficients.coefficients()) {
    h = H::combine(std::move(h), coefficient);
    ++degree;
  }
  F zero = F::Zero();
  for (size_t i = degree; i < MaxDegree + 1; ++i) {
    h = H::combine(std::move(h), zero);
  }
  return h;
}

}  // namespace math

namespace base {

template <typename F, size_t MaxDegree>
class Copyable<math::UnivariateDenseCoefficients<F, MaxDegree>> {
 public:
  static bool WriteTo(
      const math::UnivariateDenseCoefficients<F, MaxDegree>& coeffs,
      Buffer* buffer) {
    return buffer->Write(coeffs.coefficients_);
  }

  static bool ReadFrom(
      const ReadOnlyBuffer& buffer,
      math::UnivariateDenseCoefficients<F, MaxDegree>* coeffs) {
    std::vector<F> raw_coeff;
    if (!buffer.Read(&raw_coeff)) return false;
    *coeffs =
        math::UnivariateDenseCoefficients<F, MaxDegree>(std::move(raw_coeff));
    return true;
  }

  static size_t EstimateSize(
      const math::UnivariateDenseCoefficients<F, MaxDegree>& coeffs) {
    return base::EstimateSize(coeffs.coefficients_);
  }
};

template <typename F, size_t MaxDegree>
class RapidJsonValueConverter<math::UnivariateDenseCoefficients<F, MaxDegree>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(
      const math::UnivariateDenseCoefficients<F, MaxDegree>& value,
      Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "coefficients", value.coefficients(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::UnivariateDenseCoefficients<F, MaxDegree>* value,
                 std::string* error) {
    std::vector<F> coeffs;
    if (!ParseJsonElement(json_value, "coefficients", &coeffs, error))
      return false;
    *value = math::UnivariateDenseCoefficients<F, MaxDegree>(std::move(coeffs));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_DENSE_COEFFICIENTS_H_
