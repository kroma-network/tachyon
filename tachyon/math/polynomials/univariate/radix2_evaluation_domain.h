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

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

// Defines a domain over which finite field (I)FFTs can be performed. Works
// only for fields that have a large multiplicative subgroup of size that is a
// power-of-2.
template <typename F, size_t MaxDegree = size_t{1} << F::Config::kTwoAdicity>
class Radix2EvaluationDomain : public UnivariateEvaluationDomain<F, MaxDegree> {
 public:
  using Field = F;
  using Evals = UnivariateEvaluations<F, MaxDegree>;
  using DensePoly = UnivariateDensePolynomial<F, MaxDegree>;
  using SparsePoly = UnivariateSparsePolynomial<F, MaxDegree>;

  constexpr static size_t kMaxDegree = MaxDegree;
  // Factor that determines if a the degree aware FFT should be called.
  constexpr static size_t kDegreeAwareFFTThresholdFactor = 1 << 2;
  // The minimum size of a chunk at which parallelization of |Butterfly()| is
  // beneficial. This value was chosen empirically.
  constexpr static size_t kMinGapSizeForParallelization = 1 << 10;
  // The minimum size of roots of unity at which parallelization of
  // |GetRootsOfUnity()| is beneficial. This value was chosen empirically.
  constexpr static uint32_t kMinLogRootsOfUnitySizeForParallelization = 7;
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
        absl::bit_ceil(num_coeffs), SafeLog2Ceiling(num_coeffs)));
  }

  // libfqfft uses >
  // https://github.com/scipr-lab/libfqfft/blob/e0183b2cef7d4c5deb21a6eaf3fe3b586d738fe0/libfqfft/evaluation_domain/domains/basic_radix2_domain.tcc#L33
  // (See
  // https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/radix2/mod.rs#L62)
  constexpr static bool IsValidNumCoeffs(size_t num_coeffs) {
    return SafeLog2Ceiling(num_coeffs) <= F::Config::kTwoAdicity;
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

  // NOTE(TomTaehoonKim): |base::bits::Log2Ceiling(size)| returns -1 on |size|
  // is 0. This ensures that it returns non negative value even in that case.
  constexpr static uint32_t SafeLog2Ceiling(size_t size) {
    return static_cast<uint32_t>(std::max(base::bits::Log2Ceiling(size), 0));
  }

  // Degree aware FFT that runs in O(n log d) instead of O(n log n).
  // Implementation copied from libiop. (See
  // https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/radix2/fft.rs#L28)
  constexpr void DegreeAwareFFTInPlace(Evals& evals) const {
    if (!this->offset_.IsOne()) {
      this->DistributePowers(evals, this->offset_);
    }
    size_t n = this->size_;
    uint32_t log_n = this->log_size_of_group_;
    size_t num_coeffs = absl::bit_ceil(evals.evaluations_.size());
    uint32_t log_d = SafeLog2Ceiling(num_coeffs);
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
    UnivariateEvaluationDomain<F, MaxDegree>::SwapElements(evals, num_coeffs,
                                                           log_n);
    if (duplicity_of_initials > 1) {
      auto chunks = base::Chunked(evals.evaluations_, duplicity_of_initials);
      std::vector<absl::Span<F>> chunks_vector =
          base::Map(chunks.begin(), chunks.end(),
                    [](const absl::Span<F>& chunk) { return chunk; });
      OPENMP_PARALLEL_FOR(size_t i = 0; i < chunks_vector.size(); ++i) {
        const absl::Span<F>& chunks = chunks_vector[i];
        const F& v = chunks[0];
        for (size_t j = 1; j < chunks.size(); ++j) {
          chunks[j] = v;
        }
      }
    }
    size_t start_gap = duplicity_of_initials;
    OutInHelper(evals, this->group_gen_, start_gap);
  }

  constexpr void InOrderFFTInPlace(Evals& evals) const {
    if (!this->offset_.IsOne()) {
      this->DistributePowers(evals, this->offset_);
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
      this->DistributePowersAndMulByConst(poly, this->offset_inv_,
                                          this->size_inv_);
    }
  }

  constexpr void FFTHelperInPlace(Evals& evals) const {
    uint32_t log_len = static_cast<uint32_t>(base::bits::Log2Ceiling(
        static_cast<uint32_t>(evals.evaluations_.size())));
    UnivariateEvaluationDomain<F, MaxDegree>::SwapElements(
        evals, evals.evaluations_.size() - 1, log_len);
    OutInHelper(evals, this->group_gen_, 1);
  }

  // Handles doing an IFFT with handling of being in order and out of order.
  // The results here must all be divided by |poly|, which is left up to the
  // caller to do.
  constexpr void IFFTHelperInPlace(DensePoly& poly) const {
    InOutHelper(poly, this->group_gen_inv_);
    uint32_t log_len = static_cast<uint32_t>(base::bits::Log2Ceiling(
        static_cast<uint32_t>(poly.coefficients_.coefficients_.size())));
    UnivariateEvaluationDomain<F, MaxDegree>::SwapElements(
        poly, poly.coefficients_.coefficients_.size() - 1, log_len);
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
    OPENMP_PARALLEL_FOR(size_t i = 0; i <= poly_or_evals.Degree();
                        i += chunk_size) {
      // If the chunk is sufficiently big that parallelism helps,
      // we parallelize the butterfly operation within the chunk.
      if (gap > kMinGapSizeForParallelization && chunk_size < thread_nums) {
        OPENMP_PARALLEL_FOR(size_t j = 0; j < gap; ++j) {
          if (j * step < roots.size()) {
            fn(*poly_or_evals[i + j], *poly_or_evals[i + j + gap],
               roots[j * step]);
          }
        }
      } else {
        for (size_t j = 0, k = 0; j < gap && k < roots.size(); ++j, k += step) {
          fn(*poly_or_evals[i + j], *poly_or_evals[i + j + gap], roots[k]);
        }
      }
    }
  }

  // Computes the first |size| roots of unity for the entire domain.
  // e.g. for the domain [1, g, g², ..., gⁿ⁻¹}] and |size| = n / 2, it computes
  // [1, g, g², ..., g^{(n / 2) - 1}]
  constexpr std::vector<F> GetRootsOfUnity(size_t size, const F& root) const {
#if defined(TACHYON_HAS_OPENMP)
    uint32_t log_size = SafeLog2Ceiling(this->size_);
    if (log_size <= kMinLogRootsOfUnitySizeForParallelization)
#endif
      return ComputePowersSerial(size, root);
#if defined(TACHYON_HAS_OPENMP)
    size_t required_log_size = size_t{SafeLog2Ceiling(size)};
    F power = root;
    // [g, g², g⁴, g⁸, ..., g^(2^(|required_log_size|))]
    std::vector<F> log_powers =
        base::CreateVector(required_log_size, [&power]() {
          F old_value = power;
          power.SquareInPlace();
          return old_value;
        });

    // allocate the return array and start the recursion
    std::vector<F> powers =
        base::CreateVector(size_t{1} << required_log_size, F::Zero());
    GetRootsOfUnityRecursive(powers, absl::MakeConstSpan(log_powers));
    return powers;
#endif
  }

#if defined(TACHYON_HAS_OPENMP)
  constexpr void GetRootsOfUnityRecursive(
      std::vector<F>& out, const absl::Span<const F>& log_powers) const {
    CHECK_EQ(out.size(), size_t{1} << log_powers.size());
    // base case: just compute the powers sequentially,
    // g = log_powers[0], |out| = [1, g, g², ..., g^(|log_powers.size() - 1|)]
    if (log_powers.size() <=
        size_t{kMinLogRootsOfUnitySizeForParallelization}) {
      out[0] = F::One();
      for (size_t i = 1; i < out.size(); ++i) {
        out[i] = out[i - 1] * log_powers[0];
      }
      return;
    }

    // recursive case:
    // 1. split |log_powers| in half
    // |log_powers| =[g, g², g⁴, g⁸]
    size_t half_size = (1 + log_powers.size()) / 2;
    // |log_powers_lo| = [g, g²]
    absl::Span<const F> log_powers_lo = log_powers.subspan(0, half_size);
    // |log_powers_lo| = [g⁴, g⁸]
    absl::Span<const F> log_powers_hi = log_powers.subspan(half_size);
    std::vector<F> src_lo =
        base::CreateVector(1 << log_powers_lo.size(), F::Zero());
    std::vector<F> src_hi =
        base::CreateVector(1 << log_powers_hi.size(), F::Zero());

    // clang-format off
    // 2. compute each half individually
    // |src_lo| = [1, g, g², g³]
    // |src_hi| = [1, g⁴, g⁸, g¹²]
    // clang-format on
#pragma omp parallel for
    for (size_t i = 0; i < 2; ++i) {
      GetRootsOfUnityRecursive(i == 0 ? src_lo : src_hi,
                               i == 0 ? log_powers_lo : log_powers_hi);
    }

    // clang-format off
    // 3. recombine halves
    // At this point, out is a blank slice.
    // |out| = [1, g, g², g³, g⁴, g⁵, g⁶, g⁷, g⁸, ... g¹², g¹³, g¹⁴, g¹⁵]
    // clang-format on
    auto out_chunks = base::Chunked(out, src_lo.size());
    std::vector<absl::Span<F>> out_chunks_vector =
        base::Map(out_chunks.begin(), out_chunks.end(),
                  [](const absl::Span<F>& chunk) { return chunk; });
#pragma omp parallel for
    for (size_t i = 0; i < out_chunks_vector.size(); ++i) {
      const F& hi = src_hi[i];
      absl::Span<F> out_chunks = out_chunks_vector[i];
      for (size_t j = 0; j < out_chunks.size(); ++j) {
        out_chunks[j] = hi * src_lo[j];
      }
    }
  }
#endif

  constexpr void InOutHelper(DensePoly& poly, const F& root) const {
    std::vector<F> roots = GetRootsOfUnity(this->size_ / 2, root);
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
          OPENMP_PARALLEL_FOR(size_t i = 0; i < size; ++i) {
            roots[i] = roots[i * (step * 2)];
          }
          roots.erase(roots.begin() + size, roots.end());
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
    std::vector<F> roots_cache = GetRootsOfUnity(this->size_ / 2, root);
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

  constexpr static std::vector<F> ComputePowersSerial(size_t size,
                                                      const F& root) {
    return ComputePowersAndMulByConstSerial(size, root, F::One());
  }

  constexpr static std::vector<F> ComputePowersAndMulByConstSerial(
      size_t size, const F& root, const F& c) {
    F value = c;
    return base::CreateVector(
        size, [&value, root]() { return std::exchange(value, value * root); });
  }

  size_t min_num_chunks_for_compaction_ = kDefaultMinNumChunksForCompaction;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_EVALUATION_DOMAIN_H_
