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

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/adapters.h"
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
  constexpr static uint32_t kSparseTwiddleDegree = 10;
  // Factor that determines if a the degree aware FFT should be called.
  constexpr static size_t kDegreeAwareFFTThresholdFactor = 1 << 2;
  // The minimum size of a chunk at which parallelization of |Butterfly()| is
  // beneficial. This value was chosen empirically.
  constexpr static size_t kMinGapSizeForParallelization = 1 << 10;
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
    if (!this->offset_.IsOne()) {
      Base::DistributePowers(evals, this->offset_);
    }
    evals.evaluations_.resize(this->size_, F::Zero());
    BestFFT(evals.evaluations_, this->group_gen_);
    return evals;
  }

  [[nodiscard]] constexpr DensePoly IFFT(const Evals& evals) const override {
    // NOTE(chokobole): This is not a faster check any more since
    // https://github.com/kroma-network/tachyon/pull/104.
    if (evals.IsZero()) return {};

    DensePoly poly;
    poly.coefficients_.coefficients_ = evals.evaluations_;
    poly.coefficients_.coefficients_.resize(this->size_, F::Zero());
    BestFFT(poly.coefficients_.coefficients_, this->group_gen_inv_);
    if (this->offset_.IsOne()) {
      // clang-format off
      OPENMP_PARALLEL_FOR(F& coeff : poly.coefficients_.coefficients_) {
        // clang-format on
        coeff *= this->size_inv_;
      }
    } else {
      Base::DistributePowersAndMulByConst(poly, this->offset_inv_,
                                          this->size_inv_);
    }
    poly.coefficients_.RemoveHighDegreeZeros();
    return poly;
  }

  template <typename PolyOrEvals>
  void BestFFT(PolyOrEvals& poly_or_evals, const F& omega) const {
#if defined(TACHYON_HAS_OPENMP)
    uint32_t thread_nums = static_cast<uint32_t>(omp_get_max_threads());
    size_t log_split = base::bits::Log2Floor(thread_nums);
    size_t n = poly_or_evals.size();
    size_t sub_n = n >> log_split;
    size_t split_m = 1 << log_split;

    if (sub_n < split_m) {
#endif
      return SerialFFT(poly_or_evals, omega, this->log_size_of_group_);
#if defined(TACHYON_HAS_OPENMP)
    } else {
      return ParallelFFT(poly_or_evals, omega, this->log_size_of_group_);
    }
#endif
  }

  template <typename PolyOrEvals>
  static void SerialFFT(PolyOrEvals& a, const F& omega, uint32_t log_n) {
    size_t n = a.size();

    Base::SwapElements(a, n, log_n);

    uint32_t m = 1;
    for (size_t i = 0; i < log_n; ++i) {
      F w_m = omega.Pow(n / (2 * m));

      size_t k = 0;
      while (k < n) {
        F w = F::One();
        for (size_t j = 0; j < m; ++j) {
          F t = a.at(k + j + m);
          t *= w;
          a.at(k + j + m) = a.at(k + j);
          a.at(k + j + m) -= t;
          a.at(k + j) += t;
          w *= w_m;
        }

        k += 2 * m;
      }

      m *= 2;
    }
  }

#if defined(TACHYON_HAS_OPENMP)
  static void SerialSplitFFT(std::vector<F>& a,
                             const std::vector<F>& twiddle_lut,
                             size_t twiddle_scale, uint32_t log_n) {
    size_t n = a.size();
    size_t m = 1;
    for (size_t i = 0; i < log_n; ++i) {
      size_t omega_idx = twiddle_scale * n / (2 * m);
      size_t low_idx = omega_idx % (1 << kSparseTwiddleDegree);
      size_t high_idx = omega_idx >> kSparseTwiddleDegree;
      F w_m = twiddle_lut[low_idx];
      if (high_idx > 0) {
        w_m *= twiddle_lut[(1 << kSparseTwiddleDegree) + high_idx];
      }

      size_t k = 0;
      while (k < n) {
        F w = F::One();
        for (size_t j = 0; j < m; ++j) {
          F t = a.at(k + j + m);
          t *= w;
          a.at(k + j + m) = a.at(k + j);
          a.at(k + j + m) -= t;
          a.at(k + j) += t;
          w *= w_m;
        }

        k += 2 * m;
      }

      m *= 2;
    }
  }
#endif

#if defined(TACHYON_HAS_OPENMP)
  template <typename PolyOrEvals>
  static void SplitRadixFFT(absl::Span<F>& tmp, const PolyOrEvals& a,
                            const std::vector<F>& twiddle_lut, size_t n,
                            size_t sub_fft_offset, uint32_t log_split) {
    size_t split_m = 1 << log_split;
    size_t sub_n = n >> log_split;

    std::vector<F> t1 = base::CreateVector<F>(split_m, F::Zero());
    for (size_t i = 0; i < split_m; ++i) {
      size_t ridx = base::bits::BitRev(i) >> (sizeof(size_t) * 8 - log_split);
      t1.at(ridx) = a.at(i * sub_n + sub_fft_offset);
    }
    SerialSplitFFT(t1, twiddle_lut, sub_n, log_split);

    size_t sparse_degree = kSparseTwiddleDegree;
    size_t omega_idx = sub_fft_offset;
    size_t low_idx = omega_idx % (1 << sparse_degree);
    size_t high_idx = omega_idx >> sparse_degree;
    F omega = twiddle_lut.at(low_idx);
    if (high_idx > 0) {
      omega *= twiddle_lut.at((1 << sparse_degree) + high_idx);
    }
    F w_m = F::One();
    for (size_t i = 0; i < split_m; ++i) {
      t1.at(i) *= w_m;
      tmp.at(i) = t1.at(i);
      w_m *= omega;
    }
  }
#endif

#if defined(TACHYON_HAS_OPENMP)
  static std::vector<F> GenerateTwiddleLookupTable(const F& omega,
                                                   uint32_t log_n,
                                                   uint32_t sparse_degree,
                                                   bool with_last_level) {
    bool without_last_level = !with_last_level;
    bool is_lut_len_large = sparse_degree > log_n;

    // dense
    if (is_lut_len_large) {
      std::vector<F> twiddle_lut =
          base::CreateVector<F>(size_t{1} << log_n, F::Zero());
      base::Parallelize(
          twiddle_lut,
          [&](absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
            F w_n = omega.Pow(chunk_index * chunk_size);
            for (F& twiddle : chunk) {
              twiddle = w_n;
              w_n *= omega;
            }
          });
      return twiddle_lut;
    }

    // sparse
    size_t low_degree_lut_len = size_t{1} << sparse_degree;
    size_t high_degree_lut_len =
        size_t{1} << (log_n - sparse_degree - uint32_t{without_last_level});
    std::vector<F> twiddle_lut = base::CreateVector<F>(
        low_degree_lut_len + high_degree_lut_len, F::Zero());
    absl::Span<F> low_degree_lut =
        absl::MakeSpan(twiddle_lut).subspan(0, low_degree_lut_len);
    absl::Span<F> high_degree_lut =
        absl::MakeSpan(twiddle_lut).subspan(low_degree_lut_len);
    base::Parallelize(
        low_degree_lut,
        [&](absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
          F w_n = omega.Pow(chunk_index * chunk_size);
          for (F& twiddle : chunk) {
            twiddle = w_n;
            w_n *= omega;
          }
        });

    F high_degree_omega = omega.Pow(uint64_t{1} << sparse_degree);
    base::Parallelize(
        high_degree_lut,
        [&](absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
          F w_n = high_degree_omega.Pow(chunk_index * chunk_size);
          for (F& twiddle : chunk) {
            twiddle = w_n;
            w_n *= high_degree_omega;
          }
        });

    return twiddle_lut;
  }
#endif

#if defined(TACHYON_HAS_OPENMP)
  template <typename PolyOrEvals>
  static void ParallelFFT(PolyOrEvals& a, const F& omega, uint32_t log_n) {
    size_t n = a.size();
    uint32_t log_split =
        base::bits::Log2Floor(static_cast<uint32_t>(omp_get_max_threads()));
    size_t split_m = 1 << log_split;
    size_t sub_n = n >> log_split;

    std::vector<F> twiddle_lut =
        GenerateTwiddleLookupTable(omega, log_n, kSparseTwiddleDegree, true);

    // split fft
    std::vector<F> tmp = base::CreateVector<F>(n, F::Zero());
    base::ParallelizeByChunkSize(
        tmp, sub_n,
        [&](absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
          size_t split_fft_offset = chunk_index * chunk_size >> log_split;
          base::ParallelizeByChunkSize(
              chunk, split_m, [&](absl::Span<F> chunk, size_t chunk_index) {
                size_t sub_fft_offset = split_fft_offset + chunk_index;
                SplitRadixFFT(chunk, a, twiddle_lut, n, sub_fft_offset,
                              log_split);
              });
        });

    // shuffle
    base::Parallelize(
        a, [&](absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
          for (size_t k = 0; k < chunk.size(); ++k) {
            size_t idx = chunk_index * chunk_size + k;
            size_t i = idx / sub_n;
            size_t j = idx % sub_n;
            chunk.at(k) = tmp.at(j * split_m + i);
          }
        });

    // sub fft
    F new_omega = omega.Pow(split_m);
    base::ParallelizeByChunkSize(
        a, sub_n,
        [&](absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
          SerialFFT(chunk, new_omega, log_n - log_split);
        });

    // copy & unshuffle
    size_t mask = (1 << log_split) - 1;
    base::Parallelize(
        tmp, [&](absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
          for (size_t i = 0; i < chunk.size(); ++i) {
            size_t idx = chunk_index * chunk_size + i;
            chunk.at(i) = a.at(idx);
          }
        });
    base::Parallelize(
        a, [&](absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
          for (size_t i = 0; i < chunk.size(); ++i) {
            size_t idx = chunk_index * chunk_size + i;
            chunk.at(i) = tmp.at(sub_n * (idx & mask) + (idx >> log_split));
          }
        });
    for (size_t i = 0; i < a.size(); ++i) {
      a.at(i) = tmp.at(sub_n * (i & mask) + (i >> log_split));
    }
  }
#endif
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_EVALUATION_DOMAIN_H_
