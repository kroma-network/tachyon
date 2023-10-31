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

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/univariate/support_poly_operators.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_forwards.h"

namespace tachyon {
namespace math {

template <typename F, size_t N>
class UnivariateSparseCoefficients;

// DenseCoefficients class provides a representation for polynomials where
// coefficients are stored contiguously for each degree. This is efficient for
// polynomials where most of the degrees have non-zero coefficients.
template <typename F, size_t N>
class UnivariateDenseCoefficients {
 public:
  constexpr static size_t kMaxSize = N;

  using Field = F;

  constexpr UnivariateDenseCoefficients() = default;
  constexpr explicit UnivariateDenseCoefficients(
      const std::vector<F>& coefficients)
      : coefficients_(coefficients) {
    CHECK_LE(coefficients_.size(), N);
    RemoveHighDegreeZeros();
  }
  constexpr explicit UnivariateDenseCoefficients(std::vector<F>&& coefficients)
      : coefficients_(std::move(coefficients)) {
    CHECK_LE(coefficients_.size(), N);
    RemoveHighDegreeZeros();
  }

  constexpr static UnivariateDenseCoefficients Zero() {
    return UnivariateDenseCoefficients();
  }

  // NOTE(chokobole): This doesn't call |RemoveHighDegreeZeros()| internally.
  // So when the returned evaluations is called with |IsZero()|, it returns
  // false. This is only used at |EvaluationDomain|.
  constexpr static UnivariateDenseCoefficients UnsafeZero(size_t size) {
    UnivariateDenseCoefficients ret;
    ret.coefficients_ = base::CreateVector(size, F::Zero());
    return ret;
  }

  constexpr static UnivariateDenseCoefficients One() {
    return UnivariateDenseCoefficients({F::One()});
  }

  constexpr static UnivariateDenseCoefficients Random(size_t size) {
    return UnivariateDenseCoefficients(
        base::CreateVector(size, []() { return F::Random(); }));
  }

  constexpr bool operator==(const UnivariateDenseCoefficients& other) const {
    return coefficients_ == other.coefficients_;
  }

  constexpr bool operator!=(const UnivariateDenseCoefficients& other) const {
    return !operator==(other);
  }

  constexpr F* operator[](size_t i) {
    return const_cast<F*>(std::as_const(*this).operator[](i));
  }

  constexpr const F* operator[](size_t i) const {
    if (i < coefficients_.size()) {
      return &coefficients_[i];
    }
    return nullptr;
  }

  constexpr const F* GetLeadingCoefficient() const {
    if (IsZero()) return nullptr;
    return &coefficients_.back();
  }

  constexpr bool IsZero() const { return coefficients_.empty(); }

  constexpr bool IsOne() const {
    return coefficients_.size() == 1 && coefficients_[0].IsOne();
  }

  constexpr size_t Degree() const {
    if (IsZero()) return 0;
    return coefficients_.size() - 1;
  }

  constexpr F Evaluate(const F& point) const {
    if (IsZero()) return F::Zero();
    if (point.IsZero()) return coefficients_[0];
    return DoEvaluate(point);
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

  const std::vector<F>& coefficients() const { return coefficients_; }

 private:
  friend class internal::UnivariatePolynomialOp<
      UnivariateDenseCoefficients<F, N>>;
  friend class internal::UnivariatePolynomialOp<
      UnivariateSparseCoefficients<F, N>>;
  friend class Radix2EvaluationDomain<F, N>;
  friend class MixedRadixEvaluationDomain<F, N>;
  friend class base::Copyable<UnivariateDenseCoefficients<F, N>>;

  constexpr F DoEvaluate(const F& point) const {
#if defined(TACHYON_HAS_OPENMP)
    // Horner's method - parallel method
    // compute the number of threads we will be using.
    size_t thread_nums = static_cast<uint32_t>(omp_get_max_threads());
    size_t num_coeffs = coefficients_.size();
    size_t num_coeffs_per_thread = (num_coeffs + thread_nums - 1) / thread_nums;

    // run Horner's method on each thread as follows:
    // 1) Split up the coefficients across each thread evenly.
    // 2) Do polynomial evaluation via Horner's method for the thread's
    // coefficients
    // 3) Scale the result point^{thread coefficient start index}
    // Then obtain the final polynomial evaluation by summing each threads
    // result.
    auto chunks = base::Chunked(coefficients_, num_coeffs_per_thread);
    std::vector<absl::Span<const F>> chunks_vector =
        base::Map(chunks.begin(), chunks.end(),
                  [](absl::Span<const F> chunk) { return chunk; });
    std::vector<F> results =
        base::CreateVector(chunks_vector.size(), F::Zero());
#pragma omp parallel for
    for (size_t i = 0; i < chunks_vector.size(); ++i) {
      F result = HornerEvaluate(chunks_vector[i], point);
      result *= point.Pow(BigInt<1>(i * num_coeffs_per_thread));
      results[i] = std::move(result);
    }
    return std::accumulate(results.begin(), results.end(), F::Zero(),
                           std::plus<>());
#else
    return HornerEvaluate(absl::MakeConstSpan(coefficients_), point);
#endif
  }

  constexpr static F HornerEvaluate(absl::Span<const F> coefficients,
                                    const F& point) {
    return std::accumulate(coefficients.rbegin(), coefficients.rend(),
                           F::Zero(),
                           [&point](const F& result, const F& coeff) {
                             return result * point + coeff;
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

}  // namespace math

namespace base {

template <typename F, size_t N>
class Copyable<math::UnivariateDenseCoefficients<F, N>> {
 public:
  static bool WriteTo(const math::UnivariateDenseCoefficients<F, N>& coeffs,
                      Buffer* buffer) {
    return buffer->Write(coeffs.coefficients_);
  }

  static bool ReadFrom(const Buffer& buffer,
                       math::UnivariateDenseCoefficients<F, N>* coeffs) {
    std::vector<F> raw_coeff;
    if (!buffer.Read(&raw_coeff)) return false;
    *coeffs = math::UnivariateDenseCoefficients<F, N>(std::move(raw_coeff));
    return true;
  }

  static size_t EstimateSize(
      const math::UnivariateDenseCoefficients<F, N>& coeffs) {
    return base::EstimateSize(coeffs.coefficients_);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_DENSE_COEFFICIENTS_H_
