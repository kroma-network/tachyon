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
#include "tachyon/math/finite_fields/packed_prime_field_traits_forward.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/matrix_utils.h"
#include "tachyon/math/polynomials/univariate/two_adic_subgroup.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

// Defines a domain over which finite field (I)FFTs can be performed. Works
// only for fields that have a large multiplicative subgroup of size that is a
// power-of-2.
template <typename F,
          size_t MaxDegree = (size_t{1} << F::Config::kTwoAdicity) - 1>
class Radix2EvaluationDomain : public UnivariateEvaluationDomain<F, MaxDegree>,
                               public TwoAdicSubgroup<F> {
 public:
  using Base = UnivariateEvaluationDomain<F, MaxDegree>;
  using Field = F;
  using Evals = UnivariateEvaluations<F, MaxDegree>;
  using DensePoly = UnivariateDensePolynomial<F, MaxDegree>;
  using SparsePoly = UnivariateSparsePolynomial<F, MaxDegree>;
  using PackedPrimeField =
      // NOLINTNEXTLINE(whitespace/operators)
      std::conditional_t<F::Config::kModulusBits <= 32,
                         typename PackedPrimeFieldTraits<F>::PackedPrimeField,
                         F>;

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

  void FFTBatch(RowMajorMatrix<F>& mat) override {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    CHECK_EQ(this->size_, static_cast<size_t>(mat.rows()));
    size_t log_n = this->log_size_of_group_;
    mid_ = log_n / 2;

    // The first half looks like a normal DIT.
    ReverseMatrixIndexBits(mat);
    RunParallelRowChunks(mat, roots_vec_[log_n - 1], packed_roots_vec_[0]);

    // For the second half, we flip the DIT, working in bit-reversed order.
    ReverseMatrixIndexBits(mat);
    RunParallelRowChunksReversed(mat, rev_roots_vec_, packed_roots_vec_[1]);
    ReverseMatrixIndexBits(mat);
  }

  CONSTEXPR_IF_NOT_OPENMP void CosetLDEBatch(RowMajorMatrix<F>& mat,
                                             size_t added_bits) {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    CHECK_EQ(this->size_, static_cast<size_t>(mat.rows()));
    size_t log_n = this->log_size_of_group_;
    mid_ = log_n / 2;

    // The first half looks like a normal DIT.
    ReverseMatrixIndexBits(mat);
    RunParallelRowChunks(mat, inv_roots_vec_[0], packed_inv_roots_vec_[0]);

    // For the second half, we flip the DIT, working in bit-reversed order.
    ReverseMatrixIndexBits(mat);
    RunParallelRowChunksReversed(mat, rev_inv_roots_vec_,
                                 packed_inv_roots_vec_[1]);
    // We skip the final bit-reversal, since the next FFT expects bit-reversed
    // input.

    // Rescale coefficients in two ways:
    // - divide by number of rows (since we're doing an inverse DFT)
    // - multiply by powers of the coset shift (see default coset LDE impl for
    // an explanation)
    std::vector<F> weights = F::GetSuccessivePowers(
        this->size_, F::FromMontgomery(F::Config::kSubgroupGenerator),
        this->size_inv_);
    OPENMP_PARALLEL_FOR(size_t row = 0; row < weights.size(); ++row) {
      // Reverse bits because |mat| is encoded in bit-reversed order
      mat.row(base::bits::BitRev(row) >>
              (sizeof(size_t) * 8 - this->log_size_of_group_)) *= weights[row];
    }
    ExpandInPlaceWithZeroPad<RowMajorMatrix<F>>(mat, added_bits);

    size_t rows = static_cast<size_t>(mat.rows());
    CHECK(base::bits::IsPowerOfTwo(rows));
    std::unique_ptr<Radix2EvaluationDomain> domain =
        Radix2EvaluationDomain<F>::Create(rows);
    log_n = domain->log_size_of_group_;
    mid_ = log_n / 2;

    // The first half looks like a normal DIT.
    domain->RunParallelRowChunks(mat, domain->roots_vec_[log_n - 1],
                                 domain->packed_roots_vec_[0]);

    // For the second half, we flip the DIT, working in bit-reversed order.
    ReverseMatrixIndexBits(mat);
    domain->RunParallelRowChunksReversed(mat, domain->rev_roots_vec_,
                                         domain->packed_roots_vec_[1]);
    ReverseMatrixIndexBits(mat);
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

  CONSTEXPR_IF_NOT_OPENMP void InOrderIFFTInPlace(DensePoly& poly) const {
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
  CONSTEXPR_IF_NOT_OPENMP static void ApplyButterfly(PolyOrEvals& poly_or_evals,
                                                     absl::Span<const F> roots,
                                                     size_t gap) {
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
  CONSTEXPR_IF_NOT_OPENMP void PrepareRootsVecCache() {
    if (this->log_size_of_group_ == 0) return;

    roots_vec_.resize(this->log_size_of_group_);
    inv_roots_vec_.resize(this->log_size_of_group_);

    size_t vec_largest_size = this->size_ / 2;

    // Compute biggest vector of |root_vec_| and |inv_root_vec_| first.
    std::vector<F> largest =
        this->GetRootsOfUnity(vec_largest_size, this->group_gen_);
    std::vector<F> largest_inv =
        this->GetRootsOfUnity(vec_largest_size, this->group_gen_inv_);

    if constexpr (F::Config::kModulusBits <= 32) {
      packed_roots_vec_.resize(2);
      packed_inv_roots_vec_.resize(2);
      packed_roots_vec_[0].resize(vec_largest_size);
      packed_inv_roots_vec_[0].resize(vec_largest_size);
      packed_roots_vec_[1].resize(vec_largest_size);
      packed_inv_roots_vec_[1].resize(vec_largest_size);
      rev_roots_vec_ = ReverseSliceIndexBits(largest);
      rev_inv_roots_vec_ = ReverseSliceIndexBits(largest_inv);
      for (size_t i = 0; i < vec_largest_size; ++i) {
        packed_roots_vec_[0][i] = PackedPrimeField::Broadcast(largest[i]);
        packed_inv_roots_vec_[0][i] =
            PackedPrimeField::Broadcast(largest_inv[i]);
        packed_inv_roots_vec_[1][i] =
            PackedPrimeField::Broadcast(rev_roots_vec_[i]);
        packed_inv_roots_vec_[1][i] =
            PackedPrimeField::Broadcast(rev_inv_roots_vec_[i]);
      }
    }

    roots_vec_[this->log_size_of_group_ - 1] = largest;
    inv_roots_vec_[0] = largest_inv;

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

  // This can be used as the first half of a parallelized butterfly network.
  CONSTEXPR_IF_NOT_OPENMP void RunParallelRowChunks(
      RowMajorMatrix<F>& mat, const std::vector<F>& twiddles,
      const std::vector<PackedPrimeField>& packed_twiddles_rev) {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    CHECK(base::bits::IsPowerOfTwo(mat.rows()));
    size_t cols = static_cast<size_t>(mat.cols());
    size_t chunk_rows = 1 << mid_;

    // max block size: 2^|mid_|
    // TODO(ashjeong): benchmark between |OPENMP_PARALLEL_FOR| here vs
    // |OPENMP_NESTED_PARALLEL_FOR| in |RunDitLayers|
    for (size_t block_start = 0; block_start < this->size_;
         block_start += chunk_rows) {
      size_t cur_chunk_rows = std::min(chunk_rows, this->size_ - block_start);
      Eigen::Block<RowMajorMatrix<F>> submat =
          mat.block(block_start, 0, cur_chunk_rows, cols);
      for (size_t layer = 0; layer < mid_; ++layer) {
        RunDitLayers(submat, layer, absl::MakeSpan(twiddles),
                     absl::MakeSpan(packed_twiddles_rev), false);
      }
    }
  }

  // This can be used as the second half of a parallelized butterfly network.
  CONSTEXPR_IF_NOT_OPENMP void RunParallelRowChunksReversed(
      RowMajorMatrix<F>& mat, const std::vector<F>& twiddles_rev,
      const std::vector<PackedPrimeField>& packed_twiddles_rev) {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    CHECK(base::bits::IsPowerOfTwo(mat.rows()));
    size_t cols = static_cast<size_t>(mat.cols());
    size_t chunk_rows = 1 << (this->log_size_of_group_ - mid_);

    // max block size: 2^(|this->log_size_of_group_| - |mid_|)
    // TODO(ashjeong): benchmark between |OPENMP_PARALLEL_FOR| here vs
    // |OPENMP_NESTED_PARALLEL_FOR| in |RunDitLayers|
    for (size_t block_start = 0; block_start < this->size_;
         block_start += chunk_rows) {
      size_t thread = block_start / chunk_rows;
      size_t cur_chunk_rows = std::min(chunk_rows, this->size_ - block_start);
      Eigen::Block<RowMajorMatrix<F>> submat =
          mat.block(block_start, 0, cur_chunk_rows, cols);
      for (size_t layer = mid_; layer < this->log_size_of_group_; ++layer) {
        size_t first_block = thread << (layer - mid_);
        RunDitLayers(submat, layer,
                     absl::MakeSpan(twiddles_rev.data() + first_block,
                                    twiddles_rev.size() - first_block),
                     absl::MakeSpan(packed_twiddles_rev.data() + first_block,
                                    packed_twiddles_rev.size() - first_block),
                     true);
      }
    }
  }

  CONSTEXPR_IF_NOT_OPENMP void RunDitLayers(
      Eigen::Block<RowMajorMatrix<F>>& submat, size_t layer,
      const absl::Span<const F>& twiddles,
      const absl::Span<const PackedPrimeField>& packed_twiddles, bool rev) {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    size_t layer_rev = this->log_size_of_group_ - 1 - layer;
    size_t half_block_size = rev ? 1 << layer_rev : 1 << layer;
    size_t block_size = half_block_size * 2;
    size_t sub_rows = static_cast<size_t>(submat.rows());
    DCHECK_GE(sub_rows, block_size);

    OPENMP_PARALLEL_NESTED_FOR(size_t block_start = 0; block_start < sub_rows;
                               block_start += block_size) {
      for (size_t i = 0; i < half_block_size; ++i) {
        size_t lo = block_start + i;
        size_t hi = lo + half_block_size;
        const F& twiddle =
            rev ? twiddles[block_start / block_size] : twiddles[i << layer_rev];
        const PackedPrimeField& packed_twiddle =
            rev ? packed_twiddles[block_start / block_size]
                : packed_twiddles[i << layer_rev];
        ApplyButterflyToRows(submat, lo, hi, twiddle, packed_twiddle);
      }
    }
  }

  CONSTEXPR_IF_NOT_OPENMP std::vector<F> ReverseSliceIndexBits(
      const std::vector<F>& vals) {
    size_t n = vals.size();
    if (n == 0) {
      return vals;
    }
    CHECK(base::bits::IsPowerOfTwo(n));
    size_t log_n = base::bits::Log2Ceiling(n);

    std::vector<F> ret = vals;
    this->SwapElements(ret, n, log_n);
    return ret;
  }

  CONSTEXPR_IF_NOT_OPENMP void ApplyButterflyToRows(
      Eigen::Block<RowMajorMatrix<F>>& mat, size_t row_1, size_t row_2,
      const F& twiddle, const PackedPrimeField& packed_twiddle) {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    std::vector<F*> suffix_1;
    std::vector<F*> suffix_2;

    std::vector<PackedPrimeField*> shorts_1 =
        PackRowHorizontally<PackedPrimeField>(mat, row_1, suffix_1);
    std::vector<PackedPrimeField*> shorts_2 =
        PackRowHorizontally<PackedPrimeField>(mat, row_2, suffix_2);

    OPENMP_PARALLEL_FOR(size_t i = 0; i < shorts_1.size(); ++i) {
      UnivariateEvaluationDomain<F, MaxDegree>::template ButterflyFnOutIn<
          PackedPrimeField>(*shorts_1[i], *shorts_2[i], packed_twiddle);
    }
    for (size_t i = 0; i < suffix_1.size(); ++i) {
      UnivariateEvaluationDomain<F, MaxDegree>::template ButterflyFnOutIn<F>(
          *suffix_1[i], *suffix_2[i], twiddle);
    }
  }

  size_t mid_ = 0;
  // For small prime fields
  std::vector<F> rev_roots_vec_;
  std::vector<F> rev_inv_roots_vec_;
  std::vector<std::vector<PackedPrimeField>> packed_roots_vec_;
  std::vector<std::vector<PackedPrimeField>> packed_inv_roots_vec_;
  // For all finite fields
  std::vector<std::vector<F>> roots_vec_;
  std::vector<std::vector<F>> inv_roots_vec_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_EVALUATION_DOMAIN_H_
