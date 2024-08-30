// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
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
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "gtest/gtest_prod.h"
#include "third_party/eigen3/Eigen/Core"

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/profiler.h"
#include "tachyon/math/finite_fields/packed_field_traits_forward.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/matrix_utils.h"
#include "tachyon/math/polynomials/univariate/evaluations_utils.h"
#include "tachyon/math/polynomials/univariate/radix2_twiddle_cache.h"
#include "tachyon/math/polynomials/univariate/two_adic_subgroup.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

// Defines a domain over which finite field (I)FFTs can be performed. Works
// only for fields that have a large multiplicative subgroup of size that is a
// power-of-2.
template <typename F,
          size_t MaxDegree = (size_t{1} << F::Config::kTwoAdicity) - 1>
class Radix2EvaluationDomain
    : public UnivariateEvaluationDomain<F, MaxDegree>,
      public TwoAdicSubgroup<Radix2EvaluationDomain<F, MaxDegree>> {
 public:
  using Base = UnivariateEvaluationDomain<F, MaxDegree>;
  using Field = F;
  using Evals = UnivariateEvaluations<F, MaxDegree>;
  using DensePoly = UnivariateDensePolynomial<F, MaxDegree>;
  using SparsePoly = UnivariateSparsePolynomial<F, MaxDegree>;
  using PackedPrimeField =
      // NOLINTNEXTLINE(whitespace/operators)
      std::conditional_t<F::Config::kModulusBits <= 32,
                         typename PackedFieldTraits<F>::PackedField, F>;

  constexpr static size_t kMaxDegree = MaxDegree;
  // Factor that determines if a the degree aware FFT should be called.
  constexpr static size_t kDegreeAwareFFTThresholdFactor = size_t{1} << 2;

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
    ret->cache_ =
        Radix2TwiddleCache<F>::GetItem(ret.get(), /*packed_vec_only=*/false);
    return ret;
  }

  // libfqfft uses >
  // https://github.com/scipr-lab/libfqfft/blob/e0183b2/libfqfft/evaluation_domain/domains/basic_radix2_domain.tcc#L33
  // (See
  // https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/radix2/mod.rs#L62)
  constexpr static bool IsValidNumCoeffs(size_t num_coeffs) {
    return base::bits::SafeLog2Ceiling(num_coeffs) <= F::Config::kTwoAdicity;
  }

  template <typename Derived>
  void FFTBatch(Eigen::MatrixBase<Derived>& mat) const {
    TRACE_EVENT("EvaluationDomain", "Radix2EvaluationDomain::FFTBatch");
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    CHECK_GT(cache_->roots_vec.size(), size_t{0});
    CHECK_GT(cache_->packed_roots_vec.size(), size_t{0});
    CHECK_EQ(this->size_, static_cast<size_t>(mat.rows()));

    // The first half looks like a normal DIT.
    ReverseMatrixIndexBits(mat);
    RunParallelRowChunks(mat, cache_->roots_vec.back(),
                         cache_->packed_roots_vec[0]);

    // For the second half, we flip the DIT, working in bit-reversed order.
    ReverseMatrixIndexBits(mat);
    RunParallelRowChunksReversed(mat, cache_->rev_roots_vec,
                                 cache_->packed_roots_vec[1]);
    ReverseMatrixIndexBits(mat);
  }

  template <typename Derived>
  CONSTEXPR_IF_NOT_OPENMP RowMajorMatrix<F> CosetLDEBatch(
      Eigen::MatrixBase<Derived>& mat, size_t added_bits, F shift) const {
    TRACE_EVENT("EvaluationDomain", "Radix2EvaluationDomain::CosetLDEBatch");
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    CHECK_GT(cache_->roots_vec.size(), size_t{0});
    CHECK_GT(cache_->packed_roots_vec.size(), size_t{0});
    CHECK_EQ(this->size_, static_cast<size_t>(mat.rows()));

    // The first half looks like a normal DIT.
    ReverseMatrixIndexBits(mat);
    RunParallelRowChunks(mat, cache_->inv_roots_vec[0],
                         cache_->packed_inv_roots_vec[0]);

    // For the second half, we flip the DIT, working in bit-reversed order.
    ReverseMatrixIndexBits(mat);
    RunParallelRowChunksReversed(mat, cache_->rev_inv_roots_vec,
                                 cache_->packed_inv_roots_vec[1]);
    // We skip the final bit-reversal, since the next FFT expects bit-reversed
    // input.

    // Rescale coefficients in two ways:
    // - divide by number of rows (since we're doing an inverse DFT)
    // - multiply by powers of the coset shift (see default coset LDE impl for
    // an explanation)
    base::Parallelize(this->size_, [this, &mat, &shift](size_t len,
                                                        size_t chunk_offset,
                                                        size_t chunk_size) {
      // Reverse bits because |mat| is encoded in bit-reversed order
      size_t start = chunk_offset * chunk_size;
      F weight = this->size_inv_ * shift.Pow(start);
      // NOTE: It is not possible to have empty chunk so this is safe
      for (size_t row = start; row < start + len - 1; ++row) {
        mat.row(base::bits::ReverseBitsLen(row, this->log_size_of_group_)) *=
            weight;
        weight *= shift;
      }
      mat.row(base::bits::ReverseBitsLen(start + len - 1,
                                         this->log_size_of_group_)) *= weight;
    });
    RowMajorMatrix<F> ret = ExpandInPlaceWithZeroPad(mat, added_bits);

    size_t rows = static_cast<size_t>(ret.rows());
    uint32_t log_size_of_group = base::bits::CheckedLog2(rows);
    auto domain =
        absl::WrapUnique(new Radix2EvaluationDomain(rows, log_size_of_group));
    domain->cache_ =
        Radix2TwiddleCache<F>::GetItem(domain.get(), /*packed_vec_only=*/true);

    // The first half looks like a normal DIT.
    domain->RunParallelRowChunks(ret, domain->cache_->roots_vec.back(),
                                 domain->cache_->packed_roots_vec[0]);

    // For the second half, we flip the DIT, working in bit-reversed order.
    ReverseMatrixIndexBits(ret);
    domain->RunParallelRowChunksReversed(ret, domain->cache_->rev_roots_vec,
                                         domain->cache_->packed_roots_vec[1]);
    ReverseMatrixIndexBits(ret);
    return ret;
  }

 private:
  template <typename T>
  FRIEND_TEST(UnivariateEvaluationDomainTest, RootsOfUnity);

  using UnivariateEvaluationDomain<F, MaxDegree>::UnivariateEvaluationDomain;

  // UnivariateEvaluationDomain methods
  FFTAlgorithm GetAlgorithm() const override { return FFTAlgorithm::kRadix2; }

  constexpr std::unique_ptr<UnivariateEvaluationDomain<F, MaxDegree>> Clone()
      const override {
    return absl::WrapUnique(new Radix2EvaluationDomain(*this));
  }

  CONSTEXPR_IF_NOT_OPENMP void DoFFT(Evals& evals) const override {
    TRACE_EVENT("EvaluationDomain", "Radix2EvaluationDomain::DoFFT");
    DegreeAwareFFTInPlace(evals);
  }

  CONSTEXPR_IF_NOT_OPENMP void DoIFFT(DensePoly& poly) const override {
    TRACE_EVENT("EvaluationDomain", "Radix2EvaluationDomain::DoIFFT");
    poly.coefficients_.coefficients_.resize(this->size_, F::Zero());
    InOrderIFFTInPlace(poly);
    poly.coefficients_.RemoveHighDegreeZeros();
  }

  // Degree aware FFT that runs in O(n log d) instead of O(n log n).
  // Implementation copied from libiop. (See
  // https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/radix2/fft.rs#L28)
  CONSTEXPR_IF_NOT_OPENMP void DegreeAwareFFTInPlace(Evals& evals) const {
    TRACE_EVENT("EvaluationDomain", "DegreeAwareFFTInPlace");
    if (!this->offset_.IsOne()) {
      Base::DistributePowers(evals, this->offset_);
    }
    size_t n = this->size_;
    uint32_t log_n = this->log_size_of_group_;
    size_t num_coeffs = evals.evaluations_.size();
    uint32_t log_d = base::bits::SafeLog2Ceiling(num_coeffs);
    // When the polynomial is of size k * |coset|, for k < 2ⁱ, the first i
    // iterations of Cooley-Tukey are easily predictable. This is because they
    // will be combining g(ω²) + ω * h(ω²), but g or h will always refer to a
    // coefficient that is 0. Therefore those first i rounds have the effect
    // of copying the evaluations into more locations, so we handle this in
    // initialization, and reduce the number of loops that are performing
    // arithmetic. The number of times we copy each initial non-zero element
    // is as below:
    CHECK_GE(log_n, log_d);
    size_t duplicity_of_initials = size_t{1} << (log_n - log_d);
    evals.evaluations_.resize(n, F::Zero());
    SwapBitRevElementsInPlace(evals, num_coeffs, log_n);
    size_t start_gap = 1;
    if (duplicity_of_initials >= kDegreeAwareFFTThresholdFactor) {
      base::ParallelizeByChunkSize(evals.evaluations_, duplicity_of_initials,
                                   [](absl::Span<F> chunk) {
                                     const F& v = chunk[0];
                                     for (size_t j = 1; j < chunk.size(); ++j) {
                                       chunk[j] = v;
                                     }
                                   });
      start_gap = duplicity_of_initials;
    }
    OutInHelper(evals, start_gap);
  }

  CONSTEXPR_IF_NOT_OPENMP void InOrderIFFTInPlace(DensePoly& poly) const {
    TRACE_EVENT("EvaluationDomain", "InOrderIFFTInPlace");
    IFFTHelperInPlace(poly);
    if (this->offset_.IsOne()) {
      TRACE_EVENT("Subtask", "OMPMulBySizeInv");
      // clang-format off
      OMP_PARALLEL_FOR(F& val : poly.coefficients_.coefficients_) {
        // clang-format on
        val *= this->size_inv_;
      }
    } else {
      Base::DistributePowersAndMulByConst(poly, this->offset_inv_,
                                          this->size_inv_);
    }
  }

  // Handles doing an IFFT with handling of being in order and out of order.
  // The results here must all be divided by |poly|, which is left up to the
  // caller to do.
  CONSTEXPR_IF_NOT_OPENMP void IFFTHelperInPlace(DensePoly& poly) const {
    TRACE_EVENT("EvaluationDomain", "IFFTHelperInPlace");
    InOutHelper(poly);
    SwapBitRevElementsInPlace(poly, poly.coefficients_.coefficients_.size(),
                              this->log_size_of_group_);
  }

  template <FFTOrder Order, typename PolyOrEvals>
  CONSTEXPR_IF_NOT_OPENMP static void ApplyButterfly(PolyOrEvals& poly_or_evals,
                                                     absl::Span<const F> roots,
                                                     size_t gap) {
    TRACE_EVENT("EvaluationDomain", "ApplyButterfly");
    void (*fn)(F&, F&, const F&);

    if constexpr (Order == FFTOrder::kInOut) {
      fn = UnivariateEvaluationDomain<F, MaxDegree>::ButterflyFnInOut;
    } else {
      static_assert(Order == FFTOrder::kOutIn);
      fn = UnivariateEvaluationDomain<F,
                                      MaxDegree>::template ButterflyFnOutIn<F>;
    }

    // Each butterfly cluster uses 2 * |gap| positions.
    size_t chunk_size = 2 * gap;
    OMP_PARALLEL_NESTED_FOR(size_t i = 0; i < poly_or_evals.NumElements();
                            i += chunk_size) {
      for (size_t j = 0; j < gap; ++j) {
        fn(poly_or_evals.at(i + j), poly_or_evals.at(i + j + gap), roots[j]);
      }
    }
  }

  CONSTEXPR_IF_NOT_OPENMP void InOutHelper(DensePoly& poly) const {
    TRACE_EVENT("EvaluationDomain", "InOutHelper");
    size_t gap = poly.coefficients_.coefficients_.size() / 2;
    size_t idx = 0;
    while (gap > 0) {
      ApplyButterfly<FFTOrder::kInOut>(poly, cache_->inv_roots_vec[idx++], gap);
      gap /= 2;
    }
  }

  CONSTEXPR_IF_NOT_OPENMP void OutInHelper(Evals& evals,
                                           size_t start_gap) const {
    TRACE_EVENT("EvaluationDomain", "OutInHelper");
    size_t gap = start_gap;
    size_t idx = base::bits::SafeLog2Ceiling(start_gap);
    while (gap < evals.evaluations_.size()) {
      ApplyButterfly<FFTOrder::kOutIn>(evals, cache_->roots_vec[idx++], gap);
      gap *= 2;
    }
  }

  // This can be used as the first half of a parallelized butterfly network.
  template <typename Derived>
  CONSTEXPR_IF_NOT_OPENMP void RunParallelRowChunks(
      Eigen::MatrixBase<Derived>& mat, absl::Span<const F> twiddles,
      absl::Span<const PackedPrimeField> packed_twiddles_rev) const {
    TRACE_EVENT("EvaluationDomain", "RunParallelRowChunks");
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    CHECK(base::bits::IsPowerOfTwo(mat.rows()));
    size_t cols = static_cast<size_t>(mat.cols());
    uint32_t log_n = this->log_size_of_group_;
    uint32_t mid = log_n / 2;
    size_t chunk_rows = size_t{1} << mid;

    // max block size: 2^|mid|
    OMP_PARALLEL_FOR(size_t block_start = 0; block_start < this->size_;
                     block_start += chunk_rows) {
      size_t cur_chunk_rows = std::min(chunk_rows, this->size_ - block_start);
      Eigen::Block<Derived> submat =
          mat.block(block_start, 0, cur_chunk_rows, cols);
      for (uint32_t layer = 0; layer < mid; ++layer) {
        RunDitLayers(submat, layer, twiddles, packed_twiddles_rev, false);
      }
    }
  }

  // This can be used as the second half of a parallelized butterfly network.
  template <typename Derived>
  CONSTEXPR_IF_NOT_OPENMP void RunParallelRowChunksReversed(
      Eigen::MatrixBase<Derived>& mat, absl::Span<const F> twiddles_rev,
      absl::Span<const PackedPrimeField> packed_twiddles_rev) const {
    TRACE_EVENT("EvaluationDomain", "RunParallelRowChunksReversed");

    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    CHECK(base::bits::IsPowerOfTwo(mat.rows()));
    size_t cols = static_cast<size_t>(mat.cols());
    uint32_t log_n = this->log_size_of_group_;
    uint32_t mid = log_n / 2;
    size_t chunk_rows = size_t{1} << (log_n - mid);

    TRACE_EVENT("Subtask", "RunDitLayersLoop");
    // max block size: 2^(|log_n| - |mid|)
    OMP_PARALLEL_FOR(size_t block_start = 0; block_start < this->size_;
                     block_start += chunk_rows) {
      size_t thread = block_start / chunk_rows;
      size_t cur_chunk_rows = std::min(chunk_rows, this->size_ - block_start);
      Eigen::Block<Derived> submat =
          mat.block(block_start, 0, cur_chunk_rows, cols);
      for (uint32_t layer = mid; layer < log_n; ++layer) {
        size_t first_block = thread << (layer - mid);
        RunDitLayers(submat, layer,
                     twiddles_rev.subspan(first_block,
                                          twiddles_rev.size() - first_block),
                     packed_twiddles_rev.subspan(
                         first_block, packed_twiddles_rev.size() - first_block),
                     true);
      }
    }
  }

  template <typename Derived>
  CONSTEXPR_IF_NOT_OPENMP void RunDitLayers(
      Eigen::Block<Derived>& submat, uint32_t layer,
      absl::Span<const F> twiddles,
      absl::Span<const PackedPrimeField> packed_twiddles, bool rev) const {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    uint32_t layer_rev = this->log_size_of_group_ - 1 - layer;
    size_t half_block_size = size_t{1} << (rev ? layer_rev : layer);
    size_t block_size = half_block_size * 2;
    size_t sub_rows = static_cast<size_t>(submat.rows());
    DCHECK_GE(sub_rows, block_size);

    for (size_t block_start = 0; block_start < sub_rows;
         block_start += block_size) {
      for (size_t i = 0; i < half_block_size; ++i) {
        size_t lo = block_start + i;
        size_t hi = lo + half_block_size;
        F twiddle =
            rev ? twiddles[block_start / block_size] : twiddles[i << layer_rev];
        const PackedPrimeField& packed_twiddle =
            rev ? packed_twiddles[block_start / block_size]
                : packed_twiddles[i << layer_rev];
        ApplyButterflyToRows(submat, lo, hi, twiddle, packed_twiddle);
      }
    }
  }

  template <typename Derived>
  CONSTEXPR_IF_NOT_OPENMP static void ApplyButterflyToRows(
      Eigen::Block<Derived>& mat, size_t row_1, size_t row_2, F twiddle,
      const PackedPrimeField& packed_twiddle) {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    auto row_1_block = mat.row(row_1);
    auto row_2_block = mat.row(row_2);

    for (size_t i = 0; i < row_1_block.cols() / PackedPrimeField::N; ++i) {
      UnivariateEvaluationDomain<F, MaxDegree>::template ButterflyFnOutIn(
          *reinterpret_cast<PackedPrimeField*>(
              &row_1_block.data()[PackedPrimeField::N * i]),
          *reinterpret_cast<PackedPrimeField*>(
              &row_2_block.data()[PackedPrimeField::N * i]),
          packed_twiddle);
    }
    size_t remaining_start_idx =
        row_1_block.cols() / PackedPrimeField::N * PackedPrimeField::N;
    for (size_t i = remaining_start_idx;
         i < static_cast<size_t>(row_1_block.cols()); ++i) {
      UnivariateEvaluationDomain<F, MaxDegree>::template ButterflyFnOutIn(
          *reinterpret_cast<F*>(&row_1_block.data()[i]),
          *reinterpret_cast<F*>(&row_2_block.data()[i]), twiddle);
    }
  }

  typename Radix2TwiddleCache<F>::Item* cache_ = nullptr;
};

template <typename F, size_t MaxDegree>
struct TwoAdicSubgroupTraits<Radix2EvaluationDomain<F, MaxDegree>> {
  using Field = F;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_EVALUATION_DOMAIN_H_
