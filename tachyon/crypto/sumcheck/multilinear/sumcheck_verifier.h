// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_VERIFIER_H_
#define TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_VERIFIER_H_

#include <memory_resource>
#include <utility>
#include <vector>

#include "tachyon/crypto/sumcheck/multilinear/sumcheck_prover_msg.h"
#include "tachyon/crypto/sumcheck/multilinear/sumcheck_verifier_msg.h"
#include "tachyon/crypto/sumcheck/multilinear/sumcheck_verifying_key.h"

namespace tachyon::crypto {

template <typename F>
F InterpolateUniPoly(const std::pmr::vector<F>& poly,
                     const F& evaluation_point);

// Subclaim created when verifier is convinced.
template <typename F>
struct Subclaim {
  // The multi-dimensional point this multilinear evaluation is evaluated at.
  std::vector<F> point;
  // The expected evaluation.
  F expected_evaluation;
};

template <typename MLE>
class SumcheckVerifier {
 public:
  using F = typename MLE::Field;

  SumcheckVerifier() = default;
  explicit SumcheckVerifier(const SumcheckVerifyingKey& key)
      : num_variables_(key.num_variables),
        max_evaluations_(key.max_evaluations) {
    randomness_.reserve(key.num_variables);
    polynomials_received_.reserve(key.num_variables);
  }
  explicit SumcheckVerifier(SumcheckVerifyingKey&& key)
      : num_variables_(key.num_variables),
        max_evaluations_(key.max_evaluations) {
    randomness_.reserve(key.num_variables);
    polynomials_received_.reserve(key.num_variables);
  }

  // Runs a "round" given a |SumcheckProverMsg|, sampling and storing a random
  // value.
  //
  // While a true verifier round should perform actual verification, |Round()|
  // merely samples and stores random values. Verifications will be performed
  // altogether in |CheckAndGenerateSubclaim| after all prover and verifier
  // sub-rounds are subsequently finished.
  //
  // Referencing https://people.cs.georgetown.edu/jthaler/sumcheck.pdf
  // |random_value| is the challenge that will given from the Verifier
  // to the Prover aka rᵢ.
  template <size_t MaxDegree>
  SumcheckVerifierMsg<F> Round(SumcheckProverMsg<F, MaxDegree>&& prover_msg) {
    CHECK_LT(polynomials_received_.size(), num_variables_);

    const F random_value = F::Random();
    randomness_.push_back(random_value);
    polynomials_received_.push_back(
        std::move(prover_msg.evaluations.evaluations()));

    return {std::move(random_value)};
  }

  // Verifies the sumcheck phases' validity and generates a subclaim.
  //
  // If the asserted sum is correct, then the multilinear polynomial evaluated
  // at |subclaim.point| is |subclaim.expected_evaluation|. Otherwise, it is
  // highly unlikely that these two will be equal. A larger field size
  // guarantees a smaller soundness error.
  //
  // Referencing https://people.cs.georgetown.edu/jthaler/sumcheck.pdf
  // |CheckAndGenerateSubclaim()| verifies for each round i:
  //   gᵢ(0) + gᵢ(1) =  gᵢ₋₁(rᵢ₋₁)
  // Note: For the first round, gᵢ₋₁(rᵢ₋₁) = H = |LinearCombination.Combine()|
  //   Meanwhile, for the last round, gᵢ₋₁(rᵢ₋₁) = the |LinearCombination|
  //   evaluated on the multivariate point of Verifier's challenges.
  bool CheckAndGenerateSubclaim(const F& asserted_sum, Subclaim<F>* subclaim) {
    // Insufficient rounds.
    CHECK_EQ(polynomials_received_.size(), num_variables_);
    F expected = asserted_sum;

    for (size_t i = 0; i < num_variables_; ++i) {
      const std::pmr::vector<F>& evaluations = polynomials_received_[i];
      // Incorrect number of evaluations.
      CHECK_EQ(evaluations.size(), max_evaluations_ + 1);

      const F& p0 = evaluations[0];
      const F& p1 = evaluations[1];

      F sum = p0 + p1;

      // Check that prover message is consistent with the claim.
      if (sum != expected) {
        return false;
      }
      expected = InterpolateUniPoly(evaluations, randomness_[i]);
    }
    *subclaim = {std::move(randomness_), std::move(expected)};
    return true;
  }

 private:
  size_t num_variables_ = 0;
  size_t max_evaluations_ = 0;
  // A list storing the randomness sampled by the verifier at each round.
  std::vector<F> randomness_;
  // A list storing the univariate polynomial in evaluation form sent by the
  // prover at each round.
  std::vector<std::pmr::vector<F>> polynomials_received_;
};

// Interpolates the unique univariate polynomial |poly| of degree at most
// |poly.size()| - 1 = |poly_size| passing through the y-values in |poly| at
// x = 0,..., |poly[i].size()| - 1 and evaluates this polynomial at
// |evaluation_point|:
//   ∑ poly[i] * ∏ⱼ≠ᵢ(|evaluation_point| - j)/(i - j), (i = 0,1,...,|poly_size|)
template <typename F>
F InterpolateUniPoly(const std::pmr::vector<F>& poly,
                     const F& evaluation_point) {
  constexpr size_t kParallelFactor = 16;

  using BigInt = typename F::BigIntTy;

  // clang-format off
  // Calculating iteration i on X = |evaluation_point| is done like so:
  //
  //                             (|product| / |evals[i]|)
  //                                        |
  //                          |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
  // iteration i =  poly[i] * ∏ⱼ≠ᵢ(|evaluation_point| - j)/(i - j), (i = 0,1,...,|poly_size|)
  //                         |___|------------------------|______|
  //                                                         |
  //                                                     denom[i]
  //
  // More specifically...
  //
  //                                       |product|
  //                                           |
  //                         |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
  //                         (X - 0)(X - 1)(X - 2)...(X - poly_size)
  // iteration i = poly[i] * ―――――――――――――――――――――――――――――――――――――――
  //                                   (X - i) * denom[i]
  //                                  |______|
  //                                     |
  //                                |evals[i]|
  //
  // Where...
  //                                                  |denom_up|
  //                                                      |
  //             |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
  //            (|poly_size| - 1)! * ∏ -|offset_up|, (|offset_up| = 1,2,...,|poly_size| - i - 1)
  // denom[i] = ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
  //                     ∏ |offset_down|, (|offset_down| = |poly_size| - 1,...,i + 1)
  //                    |___________________________________________________________|
  //                                                  |
  //                                            |denom_down|
  //
  // More information on denom is given later.
  // clang-format on

  // Return early if 0 ≤ |evaluation_point| < |poly_size|, i.e. if the desired
  // value is already calculated (in |poly|)
  size_t poly_size = poly.size();
  CHECK_NE(poly_size, size_t{0});

  const BigInt test = evaluation_point.ToBigInt();
  if (test < BigInt(poly_size)) {
    return poly[test.smallest_limb()];
  }

  // |product| = ∏ⱼ(|evaluation_point| - j)
#if defined(TACHYON_HAS_OPENMP)
  size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
  size_t thread_nums = 1;
#endif
  thread_nums =
      ((thread_nums * kParallelFactor) <= poly_size) ? thread_nums : 1;

  size_t chunk_size = (poly_size + thread_nums - 1) / thread_nums;
  size_t num_chunks = (poly_size + chunk_size - 1) / chunk_size;

  std::vector<F> products(num_chunks, F::One());
  std::vector<F> denom_ups(num_chunks, F::One());
  std::vector<std::vector<F>> list_of_evals(num_chunks);
  OPENMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
    size_t begin = i * chunk_size;
    size_t len = (i == num_chunks - 1) ? poly_size - begin : chunk_size;
    list_of_evals[i].reserve(len);
    F check = F(begin);
    for (size_t j = begin; j < begin + len; ++j) {
      const F difference = evaluation_point - check;
      list_of_evals[i].push_back(difference);
      products[i] *= difference;
      if (j > 1) {
        denom_ups[i] *= check;
      }
      check += F::One();
    }
  }
  F product = products[0];
  F denom_up = denom_ups[0];
  std::vector<F> evals = list_of_evals[0];
  for (size_t i = 1; i < num_chunks; ++i) {
    evals.insert(evals.end(), list_of_evals[i].begin(), list_of_evals[i].end());
    product *= products[i];
    denom_up *= denom_ups[i];
  }

  // Computing denom[i] = ∏ⱼ≠ᵢ(i - j) for a given i:
  //
  // clang-format off
  // Start from the last step:
  //  denom[poly_size - 1] = (poly_size - 1) * (poly_size - 2) *... * 2 * 1
  //  aka:
  //   denom[poly_size - 1] = (poly_size - 1)!
  // The step before that is
  //  denom[poly_size - 2] = (poly_size - 2) * (poly_size - 3) * ... * 2 * 1 * -1
  // and the step before that is
  //  denom[poly_size - 3] = (poly_size - 3) * (poly_size - 4) * ... * 2 * 1 * -1 *-2
  // clang-format on
  //
  // i.e., for any i, the one before this will be derived from
  //  denom[i - 1] = - denom[i] * (|poly_size| - i) / i
  // aka:
  //  denom[i] = (|poly_size| - 1)! * ∏ -|offset_up|, (|offset_up| =
  //  1,2,...,|poly_size| - i - 1)
  //
  // denom is stored as a fraction number to reduce field divisions.
  F res = F::Zero();
  F offset_up = F::One();

  for (size_t i = poly_size - 1; i < SIZE_MAX; --i) {
    evals[i] *= denom_up;
    if (i != 0) {
      denom_up *= -offset_up;
      offset_up += F::One();
    }
  }
  CHECK(F::BatchInverseInPlace(evals));
  F denom_down = F::One();
  F offset_down = F(poly_size - 1);
  for (size_t i = poly_size - 1; i < SIZE_MAX; --i) {
    res += poly[i] * product * denom_down * evals[i];
    if (i != 0) {
      denom_down *= offset_down;
      offset_down -= F::One();
    }
  }
  return res;
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_VERIFIER_H_
