// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

// This header defines |Radix2EvaluationDomain|, an |UnivariateEvaluationDomain|
// for performing various kinds of polynomial arithmetic on top of fields that
// are FFT-friendly. |Radix2EvaluationDomain| supports FFTs of size at most
// 2^|F::Config::kTwoAdicity|.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_EVALUATION_DOMAIN_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_EVALUATION_DOMAIN_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

// Defines a domain over which finite field (I)FFTs can be performed. Works
// only for fields that have a large multiplicative subgroup of size that is a
// power-of-2.
template <typename F,
          size_t MaxDegree = (size_t{1} << F::Config::kTwoAdicity) - 1>
class Radix2EvaluationDomain : public UnivariateEvaluationDomain<F, MaxDegree> {
 public:
  using Base = UnivariateEvaluationDomain<F, MaxDegree>;
  using Field = F;
  using Evals = UnivariateEvaluations<F, MaxDegree>;
  using DensePoly = UnivariateDensePolynomial<F, MaxDegree>;
  using SparsePoly = UnivariateSparsePolynomial<F, MaxDegree>;

  constexpr static size_t kMaxDegree = MaxDegree;
  // Factor that determines if a the degree aware FFT should be called.
  constexpr static size_t kDegreeAwareFFTThresholdFactor = 1 << 2;

  enum class FFTOrder {
    // The input of the FFT must be in-order, but the output does not have to
    // be.
    kInOut,
    // The input of the FFT can be out of order, but the output must be
    // in-order.
    kOutIn
  };

  static std::unique_ptr<Radix2EvaluationDomain> Create(size_t num_coeffs) {
    auto ret = absl::WrapUnique(new Radix2EvaluationDomain(
        absl::bit_ceil(num_coeffs), base::bits::SafeLog2Ceiling(num_coeffs)));
    ret->PrepareRootsVecCache();
    return ret;
  }

  // libfqfft uses >
  // https://github.com/scipr-lab/libfqfft/blob/e0183b2cef7d4c5deb21a6eaf3fe3b586d738fe0/libfqfft/evaluation_domain/domains/basic_radix2_domain.tcc#L33
  // (See
  // https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/radix2/mod.rs#L62)
  constexpr static bool IsValidNumCoeffs(size_t num_coeffs) {
    return base::bits::SafeLog2Ceiling(num_coeffs) <= F::Config::kTwoAdicity;
  }

 private:
  template <typename T>
  FRIEND_TEST(UnivariateEvaluationDomainTest, RootsOfUnity);

  using UnivariateEvaluationDomain<F, MaxDegree>::UnivariateEvaluationDomain;

  // UnivariateEvaluationDomain methods
  constexpr std::unique_ptr<UnivariateEvaluationDomain<F, MaxDegree>> Clone()
      const override {
    return absl::WrapUnique(new Radix2EvaluationDomain(*this));
  }

  // UnivariateEvaluationDomain methods
  constexpr void DoFFT(Evals& evals) const override {
    if (evals.evaluations_.size() * kDegreeAwareFFTThresholdFactor <=
        this->size_) {
      DegreeAwareFFTInPlace(evals);
    } else {
      evals.evaluations_.resize(this->size_, F::Zero());
      InOrderFFTInPlace(evals);
    }
  }

  // UnivariateEvaluationDomain methods
  constexpr void DoIFFT(DensePoly& poly) const override {
    poly.coefficients_.coefficients_.resize(this->size_, F::Zero());
    InOrderIFFTInPlace(poly);
    poly.coefficients_.RemoveHighDegreeZeros();
  }

  // Degree aware FFT that runs in O(n log d) instead of O(n log n).
  // Implementation copied from libiop. (See
  // https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/radix2/fft.rs#L28)
  constexpr void DegreeAwareFFTInPlace(Evals& evals) const {
    if (!this->offset_.IsOne()) {
      Base::DistributePowers(evals, this->offset_);
    }
    size_t n = this->size_;
    uint32_t log_n = this->log_size_of_group_;
    size_t num_coeffs = absl::bit_ceil(evals.evaluations_.size());
    uint32_t log_d = base::bits::SafeLog2Ceiling(num_coeffs);
    // When the polynomial is of size k * |coset|, for k < 2ⁱ, the first i
    // iterations of Cooley-Tukey are easily predictable. This is because they
    // will be combining g(w²) + wh(w²), but g or h will always refer to a
    // coefficient that is 0. Therefore those first i rounds have the effect
    // of copying the evaluations into more locations, so we handle this in
    // initialization, and reduce the number of loops that are performing
    // arithmetic. The number of times we copy each initial non-zero element
    // is as below:
    CHECK_GE(log_n, log_d);
    size_t duplicity_of_initials = size_t{1} << (log_n - log_d);
    evals.evaluations_.resize(n, F::Zero());
    this->SwapElements(evals, num_coeffs, log_n);
    if (duplicity_of_initials > 1) {
      base::ParallelizeByChunkSize(evals.evaluations_, duplicity_of_initials,
                                   [](absl::Span<F> chunk) {
                                     const F& v = chunk[0];
                                     for (size_t j = 1; j < chunk.size(); ++j) {
                                       chunk[j] = v;
                                     }
                                   });
    }
    size_t start_gap = duplicity_of_initials;
    OutInHelper(evals, start_gap);
  }

  constexpr void InOrderFFTInPlace(Evals& evals) const {
    if (!this->offset_.IsOne()) {
      Base::DistributePowers(evals, this->offset_);
    }
    FFTHelperInPlace(evals);
  }

  constexpr void InOrderIFFTInPlace(DensePoly& poly) const {
    IFFTHelperInPlace(poly);
    if (this->offset_.IsOne()) {
      // clang-format off
      OPENMP_PARALLEL_FOR(F& val : poly.coefficients_.coefficients_) {
        // clang-format on
        val *= this->size_inv_;
      }
    } else {
      Base::DistributePowersAndMulByConst(poly, this->offset_inv_,
                                          this->size_inv_);
    }
  }

  constexpr void FFTHelperInPlace(Evals& evals) const {
    uint32_t log_len = static_cast<uint32_t>(base::bits::Log2Ceiling(
        static_cast<uint32_t>(evals.evaluations_.size())));
    this->SwapElements(evals, evals.evaluations_.size() - 1, log_len);
    OutInHelper(evals, 1);
  }

  // Handles doing an IFFT with handling of being in order and out of order.
  // The results here must all be divided by |poly|, which is left up to the
  // caller to do.
  constexpr void IFFTHelperInPlace(DensePoly& poly) const {
    InOutHelper(poly);
    uint32_t log_len = static_cast<uint32_t>(base::bits::Log2Ceiling(
        static_cast<uint32_t>(poly.coefficients_.coefficients_.size())));
    this->SwapElements(poly, poly.coefficients_.coefficients_.size() - 1,
                       log_len);
  }

  template <FFTOrder Order, typename PolyOrEvals>
  constexpr static void ApplyButterfly(PolyOrEvals& poly_or_evals,
                                       absl::Span<const F> roots, size_t gap) {
    void (*fn)(F&, F&, const F&);

    if constexpr (Order == FFTOrder::kInOut) {
      fn = UnivariateEvaluationDomain<F, MaxDegree>::ButterflyFnInOut;
    } else {
      static_assert(Order == FFTOrder::kOutIn);
      fn = UnivariateEvaluationDomain<F, MaxDegree>::ButterflyFnOutIn;
    }

    // Each butterfly cluster uses 2 * |gap| positions.
    size_t chunk_size = 2 * gap;
    OPENMP_PARALLEL_NESTED_FOR(size_t i = 0; i < poly_or_evals.NumElements();
                               i += chunk_size) {
      for (size_t j = 0; j < gap; ++j) {
        fn(poly_or_evals.at(i + j), poly_or_evals.at(i + j + gap), roots[j]);
      }
    }
  }

  // clang-format off
  // Precompute |roots_vec_| and |inv_roots_vec_| for |OutInHelper()| and |InOutHelper()|.
  // Here is an example where |this->size_| equals 32.
  // |root_vec_| = [
  //   [1],
  //   [1, ω⁸],
  //   [1, ω⁴, ω⁸, ω¹²],
  //   [1, ω², ω⁴, ω⁶, ω⁸, ω¹⁰, ω¹², ω¹⁴],
  //   [1, ω, ω², ω³, ω⁴, ω⁵, ω⁶, ω⁷, ω⁸, ω⁹, ω¹⁰, ω¹¹, ω¹², ω¹³, ω¹⁴, ω¹⁵],
  // ]
  // |inv_root_vec_| = [
  //   [1, ω⁻¹, ω⁻², ω⁻³, ω⁻⁴, ω⁻⁵, ω⁻⁶, ω⁻⁷, ω⁻⁸, ω⁻⁹, ω⁻¹⁰, ω⁻¹¹, ω⁻¹², ω⁻¹³, ω⁻¹⁴, ω⁻¹⁵],
  //   [1, ω⁻², ω⁻⁴, ω⁻⁶, ω⁻⁸, ω⁻¹⁰, ω⁻¹², ω⁻¹⁴],
  //   [1, ω⁻⁴, ω⁻⁸, ω⁻¹²],
  //   [1, ω⁻⁸],
  //   [1],
  // ]
  // clang-format on
  constexpr void PrepareRootsVecCache() {
    if (this->log_size_of_group_ == 0) return;

    roots_vec_.resize(this->log_size_of_group_);
    inv_roots_vec_.resize(this->log_size_of_group_);

    // Compute biggest vector of |root_vec_| and |inv_root_vec_| first.
    roots_vec_[this->log_size_of_group_ - 1] =
        this->GetRootsOfUnity(this->size_ / 2, this->group_gen_);
    inv_roots_vec_[0] =
        this->GetRootsOfUnity(this->size_ / 2, this->group_gen_inv_);

    // Prepare space in each vector for the others.
    size_t size = this->size_ / 2;
    for (size_t i = 1; i < this->log_size_of_group_; ++i) {
      size /= 2;
      roots_vec_[this->log_size_of_group_ - i - 1].resize(size);
      inv_roots_vec_[i].resize(size);
    }

    // Assign every element based on the biggest vector.
    OPENMP_PARALLEL_FOR(size_t i = 1; i < this->log_size_of_group_; ++i) {
      for (size_t j = 0; j < this->size_ / std::pow(2, i + 1); ++j) {
        size_t k = std::pow(2, i) * j;
        roots_vec_[this->log_size_of_group_ - i - 1][j] = roots_vec_.back()[k];
        inv_roots_vec_[i][j] = inv_roots_vec_.front()[k];
      }
    }
  }

  constexpr void InOutHelper(DensePoly& poly) const {
    size_t gap = poly.coefficients_.coefficients_.size() / 2;
    size_t idx = 0;
    while (gap > 0) {
      ApplyButterfly<FFTOrder::kInOut>(poly, inv_roots_vec_[idx++], gap);
      gap /= 2;
    }
  }

  constexpr void OutInHelper(Evals& evals, size_t start_gap) const {
    size_t gap = start_gap;
    size_t idx = base::bits::SafeLog2Ceiling(start_gap);
    while (gap < evals.evaluations_.size()) {
      ApplyButterfly<FFTOrder::kOutIn>(evals, roots_vec_[idx++], gap);
      gap *= 2;
    }
  }

  std::vector<std::vector<F>> roots_vec_;
  std::vector<std::vector<F>> inv_roots_vec_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_EVALUATION_DOMAIN_H_
