// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_PROVER_H_
#define TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_PROVER_H_

#include <utility>
#include <vector>

#include "tachyon/base/parallelize.h"
#include "tachyon/crypto/sumcheck/multilinear/sumcheck_prover_msg.h"
#include "tachyon/crypto/sumcheck/multilinear/sumcheck_proving_key.h"
#include "tachyon/crypto/sumcheck/multilinear/sumcheck_verifier_msg.h"
#include "tachyon/math/polynomials/multivariate/linear_combination.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::crypto {

// The prover argues for the sum of the polynomial over {0,1}^|num_variables_|.
//
// The polynomial is represented by a list of terms each containing a
// coefficient and a list of |MultilinearDenseEvaluations|. Refer to
// "tachyon/math/polynomials/multivariate/linear_combination.h" for more info.
template <typename MLE, size_t MaxDegree>
class SumcheckProver {
 public:
  constexpr static size_t kParallelFactor = 16;

  using F = typename MLE::Field;
  using Point = typename MLE::Point;
  using Term = math::LinearCombinationTerm<F>;

  SumcheckProver() = default;
  explicit SumcheckProver(const SumcheckProvingKey<MLE>& key)
      : num_variables_(key.verifying_key.num_variables),
        max_evaluations_(key.verifying_key.max_evaluations),
        terms_(key.terms),
        flattened_ml_evaluations_(key.flattened_ml_evaluations) {
    // Constants are not accepted.
    CHECK_NE(num_variables_, size_t{0});
    randomness_.reserve(num_variables_);
  }
  explicit SumcheckProver(SumcheckProvingKey<MLE>&& key)
      : num_variables_(key.verifying_key.num_variables),
        max_evaluations_(key.verifying_key.max_evaluations),
        terms_(std::move(key.terms)),
        flattened_ml_evaluations_(std::move(key.flattened_ml_evaluations)) {
    // Constants are not accepted.
    CHECK_NE(num_variables_, size_t{0});
    randomness_.reserve(num_variables_);
  }

  // Generate prover message, and proceed to next round.
  //
  // Main algorithm used is from section 3.2 of
  // [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
  //
  // Referencing https://people.cs.georgetown.edu/jthaler/sumcheck.pdf
  // v = the number of variables
  // For the first round, the Prover creates and sends:
  //   g₁(X₁) := ∑ g(X₁,x₂,...,xᵥ), (x₂,...,xᵥ)∈{0,1}ᵛ⁻¹
  // For the consequent round i, the Prover creates and sends:
  //   gᵢ(Xᵢ) = ∑ g(r₁,...,rᵢ₋₁,Xᵢ,xᵢ₊₁,...,xᵥ), (xᵢ₊₁ ,...,xᵥ)∈{0,1}ᵛ⁻¹
  // More specifically, |Prover(VerifierMsg)| fixes the front variables with
  // r₁,...,rᵢ₋₁, and |Prover()| creates the total gᵢ(Xᵢ).
  SumcheckProverMsg<F, MaxDegree> Round() {
    // Max number of prover rounds should be |num_variables_|.
    CHECK_LE(++round_, num_variables_);

    // Generate sum
    // clang-format off
    // g(x₁, x₂) = 1(1 - x₁)(1 - x₂) + 2x₁(1 - x₂) + 3(1 - x₁)x₂ + 4x₁x₂
    //
    // Creating the univariate polynomial:
    // g₁(X₁) = ∑ 1(1 - X₁)(1 - x₂) + 2X₁(1 - x₂) + 3(1 - X₁)x₂ + 4X₁x₂, x₂∈{0,1}
    //        = 1(1 - X₁) + 2X₁ + 3(1 - X₁) + 4X₁
    //        = start₀ + step₀X₁ + start₁ + step₁X₁
    //         (where start₀ = 1, step₀ = 1 = 2 - 1, start₁ = 3 and step₁ = 1 = 4 - 3)
    //        = start₀ + start₁ + (step₀ + step₁)X₁
    //
    // g₁(0) = start₀ + start₁
    // g₁(1) = start₀ + start₁ + step₀ + step₁
    // g₁(2) = start₀ + start₁ + 2step₀ + 2step₁
    // g₁(3) = start₀ + start₁ + 3step₀ + 3step₁
    //         where product_sum₀ = start₀ + start₁ and product_sumᵢ = product_sum₀ + (i - 1) * (step₀ + step₁)
    // clang-format on
#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
    size_t thread_nums = 1;
#endif
    size_t size = size_t{1} << (num_variables_ - round_);
    std::vector<std::vector<F>> evals_vec = base::ParallelizeMap(
        size,
        [this](size_t len, size_t chunk_offset, size_t chunk_size) {
          size_t begin = chunk_offset * chunk_size;
          std::vector<F> ret(max_evaluations_ + 1, F::Zero());
          std::vector<F> tmp(max_evaluations_ + 1);
          for (size_t i = begin; i < begin + len; ++i) {
            for (const Term& term : terms_) {
              std::fill(tmp.begin(), tmp.end(), term.coefficient);
              EvaluateTermPerVariable(i, tmp, term);
              for (size_t j = 0; j < ret.size(); ++j) {
                ret[j] += tmp[j];
              }
            }
          }
          return ret;
        },
        kParallelFactor * thread_nums);
    for (size_t i = 1; i < evals_vec.size(); ++i) {
      for (size_t j = 0; j < max_evaluations_ + 1; ++j) {
        evals_vec[0][j] += evals_vec[i][j];
      }
    }
    return {math::UnivariateEvaluations<F, MaxDegree>(std::move(evals_vec[0]))};
  }

  // Receive message from verifier and run a prover round.
  //
  // Referencing https://people.cs.georgetown.edu/jthaler/sumcheck.pdf
  // v = the number of variables
  // For each round i, the Prover creates and sends:
  //   gᵢ(Xᵢ) = ∑ g(r₁,...,rᵢ₋₁,Xᵢ,xᵢ₊₁,...,xᵥ), (xᵢ₊₁,...,xᵥ)∈{0,1}ᵛ⁻¹
  //
  // The first section of |Round(SumcheckVerifierMsg)| fixes r₁,...,rᵢ₋₁.
  // |Round()| subsequently creates gᵢ(Xᵢ).
  //
  // Note that the Prover's last round v will create:
  //   gᵥ(Xᵥ) = g(r₁,...,rᵥ₋₁,Xᵥ)
  SumcheckProverMsg<F, MaxDegree> Round(SumcheckVerifierMsg<F>&& v_msg) {
    // First round should be prover
    CHECK_GT(round_, size_t{0});
    randomness_.push_back(v_msg.random_value);
    Point point({std::move(v_msg.random_value)});
    for (MLE& evaluations : flattened_ml_evaluations_) {
      evaluations.FixVariablesInPlace(point);
    }
    return Round();
  }

 private:
  void EvaluateTermPerVariable(size_t j, std::vector<F>& products,
                               const Term& term) const {
    for (size_t index : term.indexes) {
      const MLE& table = flattened_ml_evaluations_[index];
      F start = table[j << 1];
      const F step = table[(j << 1) + 1] - start;
      // start, (start + step), (start + 2 * step), (start + 3 * step),...
      for (F& p : products) {
        p *= start;
        start += step;
      }
    }
  }

  // The current round number.
  size_t round_ = 0;
  // Number of variables.
  size_t num_variables_ = 0;
  // Max number of evaluations in a term.
  size_t max_evaluations_ = 0;
  // Sampled set of random values given by the verifier.
  std::vector<F> randomness_;
  // Stores the list of terms that is meant to be added together. Each
  // evaluation is represented by the index in |flattened_ml_evaluations_|.
  std::vector<Term> terms_;
  // Stores a list of multilinear evaluations in which |terms_| points to.
  std::vector<MLE> flattened_ml_evaluations_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_PROVER_H_
