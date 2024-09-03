// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_H_

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/base/optional.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/profiler.h"
#include "tachyon/base/range.h"
#include "tachyon/math/polynomials/evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/evaluations_utils.h"
#include "tachyon/math/polynomials/univariate/fft_algorithm.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_forwards.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

#if TACHYON_CUDA
#include "tachyon/math/polynomials/univariate/icicle/fft_algorithm_conversion.h"
#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt_holder.h"
#endif

namespace tachyon::math {

template <typename F, size_t MaxDegree>
class UnivariateEvaluationDomainFactory;

template <typename F, size_t MaxDegree>
class UnivariateEvaluationDomain : public EvaluationDomain<F, MaxDegree> {
 public:
  static_assert(F::HasRootOfUnity(),
                "UnivariateEvaluationDomain should have root of unity");

  using Field = F;
  using Evals = UnivariateEvaluations<F, MaxDegree>;
  using DensePoly = UnivariateDensePolynomial<F, MaxDegree>;
  using DenseCoeffs = UnivariateDenseCoefficients<F, MaxDegree>;
  using SparseCoeffs = UnivariateSparseCoefficients<F, MaxDegree>;
  using SparsePoly = UnivariateSparsePolynomial<F, MaxDegree>;
  using BigInt = typename F::BigIntTy;

  constexpr static size_t kMaxDegree = MaxDegree;

  constexpr UnivariateEvaluationDomain() = default;

  constexpr UnivariateEvaluationDomain(size_t size, uint32_t log_size_of_group)
      : size_(size), log_size_of_group_(log_size_of_group) {
    size_as_field_element_ = F::FromBigInt(BigInt(size_));
    size_inv_ = unwrap(size_as_field_element_.Inverse());

    // Compute the generator for the multiplicative subgroup.
    // It should be the |size_|-th root of unity.
    CHECK(F::GetRootOfUnity(size_, &group_gen_));
    // Check that it is indeed the |size_|-th root of unity.
    DCHECK_EQ(group_gen_.Pow(size_), F::One());
    group_gen_inv_ = unwrap(group_gen_.Inverse());
#if TACHYON_CUDA
    offset_big_int_ = F::One().ToBigInt();
#endif
  }

  virtual ~UnivariateEvaluationDomain() = default;

  constexpr static std::unique_ptr<UnivariateEvaluationDomain> Create(
      size_t num_coeffs) {
    return UnivariateEvaluationDomainFactory<F, MaxDegree>::Create(num_coeffs);
  }

  constexpr size_t size() const { return size_; }

  constexpr uint32_t log_size_of_group() const { return log_size_of_group_; }

  constexpr const F& size_as_field_element() const {
    return size_as_field_element_;
  }

  constexpr const F& size_inv() const { return size_inv_; }

  constexpr const F& group_gen() const { return group_gen_; }

  constexpr const F& group_gen_inv() const { return group_gen_inv_; }

  constexpr const F& offset() const { return offset_; }

  constexpr const F& offset_inv() const { return offset_inv_; }

  constexpr const F& offset_pow_size() const { return offset_pow_size_; }

#if TACHYON_CUDA
  void set_icicle(IcicleNTTHolder<F>* icicle) { icicle_ = icicle; }
#endif

  constexpr std::unique_ptr<UnivariateEvaluationDomain> GetCoset(
      const F& offset) const {
    std::unique_ptr<UnivariateEvaluationDomain> coset = Clone();
    coset->offset_ = offset;
    coset->offset_inv_ = unwrap(offset.Inverse());
    coset->offset_pow_size_ = offset.Pow(size_);
#if TACHYON_CUDA
    // TODO(chokobole): Remove this condition.
    if constexpr (IsIcicleNTTSupported<F>) {
      if (icicle_) {
        coset->offset_big_int_ = offset.ToBigInt();
      }
    }
#endif
    return coset;
  }

  // Sample an element that is *not* in the domain.
  constexpr F SampleElementOutsideDomain() const {
    F t = F::Random();
    while (EvaluateVanishingPolynomial(t).IsZero()) {
      t = F::Random();
    }
    return t;
  }

  template <typename T>
  constexpr T Zero() const {
    return T::Zero(size_ - 1);
  }

  template <typename T>
  constexpr T Random() const {
    return T::Random(size_ - 1);
  }

  virtual FFTAlgorithm GetAlgorithm() const = 0;

  // Compute a FFT.
  [[nodiscard]] constexpr Evals FFT(const DensePoly& poly) const {
    TRACE_EVENT("EvaluationDomain", "UnivariateEvaluationDomain::FFT");
    if (poly.IsZero()) return {};

    Evals evals;
    evals.evaluations_ = poly.coefficients_.coefficients_;
#if TACHYON_CUDA
    // TODO(chokobole): Remove this condition.
    if constexpr (IsIcicleNTTSupported<F>) {
      if (icicle_) {
        evals.evaluations_.resize(size_);
        CHECK((*icicle_)->FFT(FFTAlgorithmToIcicleNTTAlgorithm(GetAlgorithm()),
                              offset_big_int_, evals));
        return evals;
      }
    }
#endif
    DoFFT(evals);
    return evals;
  }

  // Compute a FFT.
  [[nodiscard]] constexpr Evals FFT(DensePoly&& poly) const {
    TRACE_EVENT("EvaluationDomain", "UnivariateEvaluationDomain::FFT");
    if (poly.IsZero()) return {};

    Evals evals;
    evals.evaluations_ = std::move(poly.coefficients_.coefficients_);
#if TACHYON_CUDA
    // TODO(chokobole): Remove this condition.
    if constexpr (IsIcicleNTTSupported<F>) {
      if (icicle_) {
        evals.evaluations_.resize(size_);
        CHECK((*icicle_)->FFT(FFTAlgorithmToIcicleNTTAlgorithm(GetAlgorithm()),
                              offset_big_int_, evals));
        return evals;
      }
    }
#endif
    DoFFT(evals);
    return evals;
  }

  constexpr virtual void DoFFT(Evals& evals) const = 0;

  // Compute an IFFT.
  [[nodiscard]] constexpr DensePoly IFFT(const Evals& evals) const {
    TRACE_EVENT("EvaluationDomain", "UnivariateEvaluationDomain::IFFT");
    // NOTE(chokobole): |evals.IsZero()| can be super slow!
    // See https://github.com/kroma-network/tachyon/pull/104.
    if (evals.evaluations_.empty()) return {};

    DensePoly poly;
    poly.coefficients_.coefficients_ = evals.evaluations_;
#if TACHYON_CUDA
    // TODO(chokobole): Remove this condition.
    if constexpr (IsIcicleNTTSupported<F>) {
      if (icicle_) {
        poly.coefficients_.coefficients_.resize(size_);
        CHECK((*icicle_)->IFFT(FFTAlgorithmToIcicleNTTAlgorithm(GetAlgorithm()),
                               offset_big_int_, poly));
        return poly;
      }
    }
#endif
    DoIFFT(poly);
    return poly;
  }

  // Compute an IFFT.
  [[nodiscard]] constexpr DensePoly IFFT(Evals&& evals) const {
    TRACE_EVENT("EvaluationDomain", "UnivariateEvaluationDomain::IFFT");
    // NOTE(chokobole): |evals.IsZero()| can be super slow!
    // See https://github.com/kroma-network/tachyon/pull/104.
    if (evals.evaluations_.empty()) return {};

    DensePoly poly;
    poly.coefficients_.coefficients_ = std::move(evals.evaluations_);
#if TACHYON_CUDA
    // TODO(chokobole): Remove this condition.
    if constexpr (IsIcicleNTTSupported<F>) {
      if (icicle_) {
        poly.coefficients_.coefficients_.resize(size_);
        CHECK((*icicle_)->IFFT(FFTAlgorithmToIcicleNTTAlgorithm(GetAlgorithm()),
                               offset_big_int_, poly));
        return poly;
      }
    }
#endif
    DoIFFT(poly);
    return poly;
  }

  constexpr virtual void DoIFFT(DensePoly& poly) const = 0;

  // Computes the first |size| roots of unity for the entire domain.
  // e.g. for the domain [1, g, g², ..., gⁿ⁻¹}] and |size| = n / 2, it
  // computes [1, g, g², ..., g^{(n / 2) - 1}]
  constexpr static std::vector<F> GetRootsOfUnity(size_t size, const F& root) {
    return F::GetSuccessivePowers(size, root);
  }

  // Define the following as
  // - H: The coset we are in, with generator g and offset h
  // - m: The size of the coset H
  // - Z_H: The vanishing polynomial for H.
  //        Z_H(x) = Π{i in m} (x - h * gⁱ) = xᵐ - hᵐ
  // - vᵢ: A sequence of values, where v₀ = 1 / (m * hᵐ⁻¹), and
  //       vᵢ₊₁ = g * vᵢ
  //
  // clang-format off
  //       Proof)
  //
  //       vᵢ = 1 / (h * gⁱ - h * g⁰) * ... * (h * gⁱ - h * gⁱ⁻¹) * (h * gⁱ - h * gⁱ⁺¹) * ... * (h * gⁱ - h * gᵐ⁻²) * (h * gⁱ - h * gᵐ⁻¹)
  //          = gᵐ⁻¹ / (h * gⁱ⁺¹ - h * g¹) * ... * (h * gⁱ⁺¹ - h * gⁱ) * (h * gⁱ⁺¹ - h * gⁱ⁺²) * ... * (h * gⁱ⁺¹ - h * gᵐ⁻¹) * (h * gⁱ⁺¹ - h * gᵐ)
  //          = gᵐ⁻¹ / (h * gⁱ⁺¹ - h * g⁰) * (h * gⁱ⁺¹ - h * g¹) * ... * (h * gⁱ⁺¹ - h * gⁱ) * (h * gⁱ⁺¹ - h * gⁱ⁺²) * ... * (h * gⁱ⁺¹ - h * gᵐ⁻¹)
  //          = gᵐ⁻¹ * vᵢ₊₁
  //          = 1 / g * vᵢ₊₁
  //
  //       v₀ = 1 / ((h - h * g¹) * (h - h * g²) * .... * (h - h * g^((m / 2) - 2)) * (h - h * g^((m / 2) - 1)) * (h - h * g^(m / 2)) * (h - h * g^((m / 2) + 1)) * (h - h * g^((m / 2) + 2))... * (h - h * gᵐ⁻²) * (h - h * gᵐ⁻¹))
  //          = 1 / (hᵐ⁻¹ * (1 - g¹) * (1 - g²) * .... * (1 - g^((m / 2) - 2)) * (1 - g^((m / 2) - 1)) * (1 - g^(m / 2)) * (1 - g^((m / 2) + 1)) * (1 - g^((m / 2) + 2))... * (1 - gᵐ⁻²) * (1 - gᵐ⁻¹))
  //          = 1 / (hᵐ⁻¹ * (1 - g¹) * (1 - g²) * .... * (1 - g^((m / 2) - 2)) * (1 - g^((m / 2) - 1)) * (1 + 1) * ... * (1 + g¹) * (1 + g²) * ... * (1 + g^((m / 2) - 2)) * (1 + g^((m / 2) - 1))) <- g^(m / 2) = -1
  //          = 1 / (2 * hᵐ⁻¹ * (1 - g¹) * (1 + g¹) * (1 - g²) * (1 + g²) * .... * (1 - g^((m / 2) - 1)) * (1 + g^((m / 2) - 1)))
  //          = 1 / (2 * hᵐ⁻¹ * (1 - g²) * (1 - g⁴) * .... * (1 - gᵐ⁻²))
  //          = 1 / (4 * hᵐ⁻¹ * (1 - g⁴) * (1 - g⁸) * .... * (1 - gᵐ⁻³))
  //          = 1 / (8 * hᵐ⁻¹ * (1 - g⁸) * (1 - g¹⁶) * .... * (1 - gᵐ⁻⁴))
  //          ...
  //          = 1 / (m * hᵐ⁻¹)
  //
  // clang-format on
  //       See Barycentric Weight for more details.
  //       https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
  // - Lᵢ_H: The value of i-th lagrange coefficient for H
  //
  // Evaluate all the lagrange polynomials defined by H at the point τ. This
  // is computed in time O(m). Then given the evaluations of a degree d
  // polynomial P over H, where d < m, P(τ) can be computed as P(τ) =
  // Σ{i in m} Lᵢ_H(τ) * P(gⁱ).
  constexpr std::vector<F> EvaluateAllLagrangeCoefficients(const F& tau) const {
    return EvaluatePartialLagrangeCoefficients(
        tau, base::Range<size_t>::Until(size_));
  }

  // Almost same with above, but it only computes parts of the lagrange
  // coefficients defined by |range|.
  // TODO(chokobole): If we want to accept IsStartInclusive as a template
  // parameter, we need a way to get a starting index from |range|.
  template <typename T, bool IsEndInclusive>
  constexpr std::vector<F> EvaluatePartialLagrangeCoefficients(
      const F& tau, base::Range<T, true, IsEndInclusive> range) const {
    TRACE_EVENT(
        "EvaluationDomain",
        "UnivariateEvaluationDomain::EvaluatePartialLagranceCoefficients");

    size_t size = range.GetSize();
    CHECK_LE(size, size_);
    if (size == 0) return {};

    // Evaluate all Lagrange polynomials at τ to get the lagrange
    // coefficients.
    //
    // We then compute Lᵢ_H(τ) as Lᵢ_H(τ) = Z_H(τ) * vᵢ / (τ - h * gⁱ)
    //
    // However, if τ is in H, both the numerator and denominator equal 0
    // when i corresponds to the value τ equals, and the coefficient is 0
    // everywhere else. We handle this case separately, and we can easily
    // detect by checking if the vanishing poly evaluates to 0 at τ.
    F z_h_at_tau = EvaluateVanishingPolynomial(tau);
    if (z_h_at_tau.IsZero()) {
      // In this case, we know that τ = h * gⁱ, for some value i.
      // Then i-th lagrange coefficient in this case is then simply 1,
      // and all other lagrange coefficients are 0.
      // Thus we find i by brute force.
      std::vector<F> u(size, F::Zero());
      F omega_i = GetElement(range.from);
      for (F& u_i : u) {
        if (omega_i == tau) {
          u_i = F::One();
          break;
        }
        omega_i *= group_gen_;
      }
      return u;
    } else {
      // In this case we have to compute Z_H(τ) * vᵢ / (τ - h * gⁱ)
      // for i in 0..|size_|. We actually compute this by computing
      // (Z_H(τ) * vᵢ)⁻¹ * (τ - h * gⁱ) and then batch inverting to
      // get the correct lagrange coefficients. We let
      // lᵢ = (Z_H(τ) * vᵢ)⁻¹ and rᵢ = τ - h * gⁱ. Notice that
      // since Z_H(τ) is i-independent, and vᵢ = g * vᵢ₋₁, it follows
      // that lᵢ = g⁻¹ * lᵢ₋₁

      // t = m * hᵐ = v₀⁻¹ * h
      F t = size_as_field_element_ * offset_pow_size_;
      F omega_i = GetElement(range.from);
      // lᵢ = (Z_H(τ) * h * gᵢ)⁻¹ * t
      //    = (Z_H(τ) * h * gᵢ * t⁻¹)⁻¹
      //    = (Z_H(τ) * h * gᵢ * v₀⁻¹ * h⁻¹)⁻¹
      //    = (Z_H(τ) * gᵢ * v₀)⁻¹
      F l_i = unwrap((z_h_at_tau * omega_i).Inverse()) * t;
      F negative_omega_i = -omega_i;
      std::vector<F> lagrange_coefficients_inverse(size);
      base::Parallelize(
          lagrange_coefficients_inverse,
          [this, &l_i, &tau, &negative_omega_i](
              absl::Span<F> chunk, size_t chunk_idx, size_t chunk_size) {
            size_t n = chunk_idx * chunk_size;
            F l_i_pow = l_i * group_gen_inv_.Pow(n);
            F negative_omega_i_pow = negative_omega_i * group_gen_.Pow(n);

            // NOTE: It is not possible to have empty chunk so this is safe
            for (size_t i = 0; i < chunk.size() - 1; ++i) {
              // (Z_H(τ) * vᵢ)⁻¹ * (τ - h * gⁱ)
              chunk[i] = l_i_pow * (tau + negative_omega_i_pow);
              // lᵢ₊₁ = g⁻¹ * lᵢ
              l_i_pow *= group_gen_inv_;
              // (- h * gⁱ) * g
              negative_omega_i_pow *= group_gen_;
            }
            chunk.back() = std::move(l_i_pow);
            chunk.back() *= tau + negative_omega_i_pow;
          });

      // Invert |lagrange_coefficients_inverse| to get the actual
      // coefficients, and return these Z_H(τ) * vᵢ / (τ - h * gⁱ)
      CHECK(F::BatchInverseInPlace(lagrange_coefficients_inverse));
      return lagrange_coefficients_inverse;
    }
  }

  // Return the sparse vanishing polynomial.
  constexpr SparsePoly GetVanishingPolynomial() const {
    TRACE_EVENT("EvaluationDomain",
                "UnivariateEvaluationDomain::GetVanishingPolynomial");
    // Z_H(x) = Π{i in m} (x - h * gⁱ) = xᵐ - hᵐ,
    // where m = |size_| and hᵐ = |offset_pow_size_|.
    return SparsePoly(
        SparseCoeffs({{0, -offset_pow_size_}, {size_, F::One()}}));
  }

  // This evaluates the vanishing polynomial for this domain at tau.
  // TODO(TomTaehoonKim): Consider precomputed exponentiation tables if we
  // need this to be faster. (See
  // https://github.com/arkworks-rs/algebra/blob/4152c41/poly/src/domain/mod.rs#L232-L233)
  constexpr F EvaluateVanishingPolynomial(const F& tau) const {
    TRACE_EVENT("EvaluationDomain",
                "UnivariateEvaluationDomain::EvaluateVanishingPolynomial");
    // Z_H(τ) = Π{i in m} (τ - h * gⁱ) = τᵐ - hᵐ,
    // where m = |size_| and hᵐ = |offset_pow_size_|.
    return tau.Pow(size_) - offset_pow_size_;
  }

  // Return the filter polynomial of |*this| with respect to |subdomain|.
  // Assumes that |subdomain| is contained within |*this|.
  //
  // Crashes if |subdomain| is not contained within |*this|.
  constexpr DensePoly GetFilterPolynomial(
      const UnivariateEvaluationDomain& subdomain) const {
    TRACE_EVENT("EvaluationDomain",
                "UnivariateEvaluationDomain::GetFilterPolynomial");
    SparsePoly domain_vanishing_poly =
        GetVanishingPolynomial() *
        SparsePoly(SparseCoeffs({{0, subdomain.size_as_field_element_ *
                                         subdomain.offset_pow_size_}}));
    SparsePoly subdomain_vanishing_poly =
        subdomain.GetVanishingPolynomial() *
        SparsePoly(SparseCoeffs({{0, size_as_field_element_}}));
    std::optional<DivResult<DensePoly>> result =
        domain_vanishing_poly.DivMod(subdomain_vanishing_poly);
    CHECK(result);
    CHECK(result->remainder.IsZero());
    return result->quotient;
  }

  // This evaluates at |tau| the filter polynomial for |*this| with respect to
  // |subdomain|.
  constexpr std::optional<F> EvaluateFilterPolynomial(
      const UnivariateEvaluationDomain& subdomain, const F& tau) const {
    TRACE_EVENT("EvaluationDomain",
                "UnivariateEvaluationDomain::EvaluateFilterPolynomial");
    F v_subdomain_of_tau = subdomain.EvaluateVanishingPolynomial(tau);
    if (v_subdomain_of_tau.IsZero()) {
      return F::One();
    } else {
      const std::optional<F> div =
          EvaluateVanishingPolynomial(tau) /
          (size_as_field_element_ * v_subdomain_of_tau);
      if (LIKELY(div)) {
        return subdomain.size_as_field_element_ * *div;
      }
      return std::nullopt;
    }
  }

  // Returns the |i|-th element of the domain.
  constexpr F GetElement(int64_t i) const {
    F result;
    if (i > 0) {
      result = group_gen_.Pow(i);
    } else {
      result = group_gen_inv_.Pow(-i);
    }
    if (!offset_.IsOne()) {
      result *= offset_;
    }
    return result;
  }

  // Returns all the elements of the domain.
  CONSTEXPR_IF_NOT_OPENMP std::vector<F> GetElements() const {
    return F::GetSuccessivePowers(size_, group_gen_, offset_);
  }

  // Multiply the i-th element of |poly_or_evals| with |g|ⁱ.
  template <typename PolyOrEvals>
  CONSTEXPR_IF_NOT_OPENMP static void DistributePowers(
      PolyOrEvals& poly_or_evals, const F& g) {
    DistributePowersAndMulByConst(poly_or_evals, g, F::One());
  }

 protected:
  // Multiply the i-th element of |poly_or_evals| with |c|*|g|ⁱ.
  template <typename PolyOrEvals>
  CONSTEXPR_IF_NOT_OPENMP static void DistributePowersAndMulByConst(
      PolyOrEvals& poly_or_evals, const F& g, const F& c) {
    TRACE_EVENT("EvaluationDomain",
                "UnivariateEvaluationDomain::DistributePowersAndMulByConst");
    // Invariant: |pow| = |c|*|g|ⁱ at the i-th iteration of the loop
    constexpr size_t kParallelFactor = 1024;
#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
    size_t thread_nums = 1;
#endif
    size_t size = poly_or_evals.NumElements();
    base::Parallelize(
        size,
        [&poly_or_evals, &g, &c](size_t len, size_t chunk_offset,
                                 size_t chunk_size) {
          TRACE_EVENT("Subtask", "MultiplyLoop");
          size_t begin = chunk_offset * chunk_size;
          F pow = c * g.Pow(begin);
          for (size_t i = begin; i < begin + len; ++i) {
            poly_or_evals.at(i) *= pow;
            pow *= g;
          }
        },
        kParallelFactor * thread_nums);
  }

  // See https://en.wikipedia.org/wiki/Butterfly_diagram
  // |lo| = |lo| + |hi|
  // |hi| = (|lo| - |hi|) * |root|
  // The simplest example would be
  // clang-format off
  // | f(ω⁰) | = | 1  0  1  0  | * | 1  0  1  0  | * | c₀ |
  // | f(ω²) |   | ω⁰ 0 -ω⁰ 0  |   | ω⁰ 0 -ω⁰ 0  |   | c₁ |
  // | f(ω¹) |   | 0  1  0  1  |   | 0  1  0  1  |   | c₂ |
  // | f(ω³) |   | 0  ω⁰ 0 -ω⁰ |   | 0  ω¹ 0 -ω¹ |   | c₃ |
  //           = | 1  0  1  0  | * | c₀      + c₂      |
  //             | ω⁰ 0 -ω⁰ 0  |   | c₀ * ω⁰ - c₂ * ω⁰ |
  //             | 0  1  0  1  |   | c₁      + c₃      |
  //             | 0  ω⁰ 0 -ω⁰ |   | c₁ * ω¹ - c₃ * ω¹ |
  //           = | c₀ + c₂                 + c₁ + c₃                  |
  //             | ω⁰ * (c₀ + c₂)          - ω⁰ * (c₁ + c₃)           |
  //             | c₀ * ω⁰ - c₂ * ω⁰       + c₁ * ω¹ - c₃ * ω¹        |
  //             | ω⁰ * (c₀ * ω⁰ - c₂ * ω⁰) -ω⁰ * (c₁ * ω¹ - c₃ * ω¹) |
  //           = | c₀ * ω⁰ + c₁ * ω⁰ + c₂ * ω⁰ + c₃ * ω⁰ |
  //             | c₀ * ω⁰ - c₁ * ω⁰ + c₂ * ω⁰ - c₃ * ω⁰ |
  //             | c₀ * ω⁰ + c₁ * ω¹ - c₂ * ω⁰ - c₃ * ω¹ |
  //             | c₀ * ω⁰ - c₁ * ω¹ - c₂ * ω⁰ + c₃ * ω¹ |
  //           = | c₀ * ω⁰ + c₁ * ω⁰ + c₂ * ω⁰ + c₃ * ω⁰ |
  //             | c₀ * ω⁰ + c₁ * ω² + c₂ * ω⁴ + c₃ * ω⁶ |
  //             | c₀ * ω⁰ + c₁ * ω¹ + c₂ * ω² + c₃ * ω³ |
  //             | c₀ * ω⁰ + c₁ * ω³ + c₂ * ω⁶ + c₃ * ω⁹ |
  // Note that the coefficients are in order and evaluations are out of order(should be swapped after).
  // clang-format on
  constexpr static void ButterflyFnInOut(F& lo, F& hi, const F& root) {
    F neg = lo - hi;

    lo += hi;

    hi = neg * root;
  }

  // See https://en.wikipedia.org/wiki/Butterfly_diagram
  // |lo| = |lo| + |hi| * |root|
  // |hi| = |lo| - |hi| * |root|
  // The simplest example would be
  // clang-format off
  // | f(ω⁰) | = | 1 0  ω⁰ 0 | * | 1  ω⁰ 0  0  | * | c₀ |
  // | f(ω¹) |   | 0 1  0  ω¹|   | 1 -ω⁰ 0  0  |   | c₂ |
  // | f(ω²) |   | 1 0 -ω⁰ 0 |   | 0  0  1  ω⁰ |   | c₁ |
  // | f(ω³) |   | 0 1  0 -ω¹|   | 0  0  1 -ω⁰ |   | c₃ |
  //           = | 1 0  ω⁰ 0 | * | c₀ + c₂ * ω⁰ |
  //             | 0 1  0  ω¹|   | c₀ - c₂ * ω⁰ |
  //             | 1 0 -ω⁰ 0 |   | c₁ + c₃ * ω⁰ |
  //             | 0 1  0 -ω¹|   | c₁ - c₃ * ω⁰ |
  //           = | 1 0  ω⁰ 0 | * | c₀ + c₂      |
  //             | 0 1  0  ω¹|   | c₀ + c₂ * ω² | ω² = -1, because ω⁴ = 1 => (ω² - 1)(ω² + 1)  = 0 => ω² = -1
  //             | 1 0 -ω⁰ 0 |   | c₁ + c₃      | Since ω is 4-th root of unity, ω² can't be 1)
  //             | 0 1  0 -ω¹|   | c₁ + c₃ * ω² |
  //           = | c₀ + c₂      + ω⁰ * (c₁ + c₃)      |
  //             | c₀ + c₂ * ω² + ω¹ * (c₁ + c₃ * ω²) |
  //             | c₀ + c₂      - ω⁰ * (c₁ + c₃)      |
  //             | c₀ + c₂ * ω² - ω¹ * (c₁ + c₃ * ω²) |
  //           = | c₀ * ω⁰ + c₁ * ω⁰ + c₂ * ω⁰ + c₃ * ω⁰ |
  //             | c₀ * ω⁰ + c₁ * ω¹ + c₂ * ω² + c₃ * ω³ |
  //             | c₀ * ω⁰ - c₁ * ω⁰ + c₂ * ω⁰ - c₃ * ω⁰ |
  //             | c₀ * ω⁰ - c₁ * ω¹ + c₂ * ω² - c₃ * ω³ |
  //           = | c₀ * ω⁰ + c₁ * ω⁰ + c₂ * ω⁰ + c₃ * ω⁰ |
  //             | c₀ * ω⁰ + c₁ * ω¹ + c₂ * ω² + c₃ * ω³ |
  //             | c₀ * ω⁰ + c₁ * ω² + c₂ * ω⁴ + c₃ * ω⁶ |
  //             | c₀ * ω⁰ + c₁ * ω³ + c₂ * ω⁶ + c₃ * ω⁹ |
  // Note that the coefficients are out of order the evaluations are in order(should be swapped before).
  // clang-format on
  template <typename FTy>
  constexpr static void ButterflyFnOutIn(FTy& lo, FTy& hi, const FTy& root) {
    hi *= root;

    FTy neg = lo - hi;

    lo += hi;

    hi = std::move(neg);
  }

  constexpr virtual std::unique_ptr<UnivariateEvaluationDomain> Clone()
      const = 0;

  // The size of the domain.
  size_t size_ = 0;
  // log2(|size_|).
  uint32_t log_size_of_group_ = 0;
  // Size of the domain as a field element.
  F size_as_field_element_;
  // Inverse of the size in the field.
  F size_inv_;
  // A generator of the subgroup.
  F group_gen_;
  // Inverse of the generator of the subgroup.
  F group_gen_inv_;
  // Offset that specifies the coset.
  F offset_ = F::One();
  // Inverse of the offset that specifies the coset.
  F offset_inv_ = F::One();
  // Constant coefficient for the vanishing polynomial.
  // Equals |offset_|^|size_|.
  F offset_pow_size_ = F::One();
#if TACHYON_CUDA
  // not owned
  const IcicleNTTHolder<F>* icicle_ = nullptr;
  BigInt offset_big_int_ = BigInt::One();
#endif
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_H_
