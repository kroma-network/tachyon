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
  // The minimum number of chunks at which root compaction is beneficial.
  constexpr static size_t kDefaultMinNumChunksForCompaction = 1 << 7;

  enum class FFTOrder {
    // The input of the FFT must be in-order, but the output does not have to
    // be.
    kInOut,
    // The input of the FFT can be out of order, but the output must be
    // in-order.
    kOutIn
  };

  static std::unique_ptr<Radix2EvaluationDomain> Create(size_t num_coeffs) {
    return absl::WrapUnique(new Radix2EvaluationDomain(
        absl::bit_ceil(num_coeffs), base::bits::SafeLog2Ceiling(num_coeffs)));
  }

  // libfqfft uses >
  // https://github.com/scipr-lab/libfqfft/blob/e0183b2cef7d4c5deb21a6eaf3fe3b586d738fe0/libfqfft/evaluation_domain/domains/basic_radix2_domain.tcc#L33
  // (See
  // https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/radix2/mod.rs#L62)
  constexpr static bool IsValidNumCoeffs(size_t num_coeffs) {
    return base::bits::SafeLog2Ceiling(num_coeffs) <= F::Config::kTwoAdicity;
  }

  void set_min_num_chunks_for_compaction(size_t min_num_chunks_for_compaction) {
    min_num_chunks_for_compaction_ = min_num_chunks_for_compaction;
  }

  size_t min_num_chunks_for_compaction() const {
    return min_num_chunks_for_compaction_;
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

  [[nodiscard]] constexpr Evals FFT(const DensePoly& poly) const override {
    if (poly.IsZero()) return {};

    Evals evals;
    evals.evaluations_ = poly.coefficients_.coefficients_;
    if (evals.evaluations_.size() * kDegreeAwareFFTThresholdFactor <=
        this->size_) {
      DegreeAwareFFTInPlace(evals);
    } else {
      evals.evaluations_.resize(this->size_, F::Zero());
      InOrderFFTInPlace(evals);
    }
    return evals;
  }

  [[nodiscard]] constexpr DensePoly IFFT(const Evals& evals) const override {
    // NOTE(chokobole): This is not a faster check any more since
    // https://github.com/kroma-network/tachyon/pull/104.
    if (evals.IsZero()) return {};

    DensePoly poly;
    poly.coefficients_.coefficients_ = evals.evaluations_;
    poly.coefficients_.coefficients_.resize(this->size_, F::Zero());
    InOrderIFFTInPlace(poly);
    poly.coefficients_.RemoveHighDegreeZeros();
    return poly;
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
    OutInHelper(evals, this->group_gen_, start_gap);
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
    OutInHelper(evals, this->group_gen_, 1);
  }

  // Handles doing an IFFT with handling of being in order and out of order.
  // The results here must all be divided by |poly|, which is left up to the
  // caller to do.
  constexpr void IFFTHelperInPlace(DensePoly& poly) const {
    InOutHelper(poly, this->group_gen_inv_);
    uint32_t log_len = static_cast<uint32_t>(base::bits::Log2Ceiling(
        static_cast<uint32_t>(poly.coefficients_.coefficients_.size())));
    this->SwapElements(poly, poly.coefficients_.coefficients_.size() - 1,
                       log_len);
  }

  template <FFTOrder Order, typename PolyOrEvals>
  constexpr static void ApplyButterfly(PolyOrEvals& poly_or_evals,
                                       absl::Span<const F> roots, size_t step,
                                       size_t chunk_size, size_t thread_nums,
                                       size_t gap) {
    void (*fn)(F&, F&, const F&);

    if constexpr (Order == FFTOrder::kInOut) {
      fn = UnivariateEvaluationDomain<F, MaxDegree>::ButterflyFnInOut;
    } else {
      static_assert(Order == FFTOrder::kOutIn);
      fn = UnivariateEvaluationDomain<F, MaxDegree>::ButterflyFnOutIn;
    }
    OPENMP_PARALLEL_NESTED_FOR(size_t i = 0; i < poly_or_evals.NumElements();
                               i += chunk_size) {
      // If the chunk is sufficiently big that parallelism helps,
      // we parallelize the butterfly operation within the chunk.
      for (size_t j = 0; j < gap; ++j) {
        if (j * step < roots.size()) {
          fn(poly_or_evals.at(i + j), poly_or_evals.at(i + j + gap),
             roots[j * step]);
        }
      }
    }
  }

  constexpr void InOutHelper(DensePoly& poly, const F& root) const {
    std::vector<F> roots = this->GetRootsOfUnity(this->size_ / 2, root);
    size_t step = 1;
    bool first = true;

#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
    size_t thread_nums = 1;
#endif

    size_t gap = poly.coefficients_.coefficients_.size() / 2;
    while (gap > 0) {
      // Each butterfly cluster uses 2 * |gap| positions.
      size_t chunk_size = 2 * gap;
      size_t num_chunks = poly.coefficients_.coefficients_.size() / chunk_size;

      // Only compact roots to achieve cache locality/compactness if the roots
      // lookup is done a significant amount of times, which also implies a
      // large lookup stride.
      bool should_compact = num_chunks >= min_num_chunks_for_compaction_;
      if (should_compact) {
        if (!first) {
          size_t size = roots.size() / (step * 2);
#if defined(TACHYON_HAS_OPENMP)
          std::vector<F> new_roots(size);
          OPENMP_PARALLEL_FOR(size_t i = 0; i < size; ++i) {
            new_roots[i] = roots[i * (step * 2)];
          }
          roots = std::move(new_roots);
#else
          for (size_t i = 0; i < size; ++i) {
            roots[i] = roots[i * (step * 2)];
          }
          roots.erase(roots.begin() + size, roots.end());
#endif
        }
        step = 1;
      } else {
        step = num_chunks;
      }
      first = false;

      ApplyButterfly<FFTOrder::kInOut>(poly, roots, step, chunk_size,
                                       thread_nums, gap);
      gap /= 2;
    }
  }

  constexpr void OutInHelper(Evals& evals, const F& root,
                             size_t start_gap) const {
    std::vector<F> roots_cache = this->GetRootsOfUnity(this->size_ / 2, root);
    // The |std::min| is only necessary for the case where
    // |min_num_chunks_for_compaction_ = 1|. Else, notice that we compact the
    // |roots_cache| by a |step| of at least |min_num_chunks_for_compaction_|.
    size_t compaction_max_size =
        std::min(roots_cache.size() / 2,
                 roots_cache.size() / min_num_chunks_for_compaction_);
    std::vector<F> compacted_roots(compaction_max_size, F::Zero());

#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
    size_t thread_nums = 1;
#endif

    size_t gap = start_gap;
    while (gap < evals.evaluations_.size()) {
      // Each butterfly cluster uses 2 * |gap| positions
      size_t chunk_size = 2 * gap;
      size_t num_chunks = evals.evaluations_.size() / chunk_size;

      // Only compact |roots| to achieve cache locality/compactness if the
      // |roots| lookup is done a significant amount of times, which also
      // implies a large lookup |step|.
      bool should_compact = num_chunks >= min_num_chunks_for_compaction_ &&
                            gap < evals.evaluations_.size() / 2;
      if (should_compact) {
        OPENMP_PARALLEL_FOR(size_t i = 0; i < gap; ++i) {
          compacted_roots[i] = roots_cache[i * num_chunks];
        }
      }
      ApplyButterfly<FFTOrder::kOutIn>(
          evals,
          should_compact ? absl::Span<const F>(compacted_roots.data(), gap)
                         : roots_cache,
          /*step=*/should_compact ? 1 : num_chunks, chunk_size, thread_nums,
          gap);
      gap *= 2;
    }
  }

  size_t min_num_chunks_for_compaction_ = kDefaultMinNumChunksForCompaction;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_EVALUATION_DOMAIN_H_
