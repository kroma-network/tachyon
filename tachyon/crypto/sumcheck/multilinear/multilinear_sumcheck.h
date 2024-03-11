// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_MULTILINEAR_SUMCHECK_H_
#define TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_MULTILINEAR_SUMCHECK_H_

#include <utility>

#include "gtest/gtest.h"

#include "tachyon/crypto/sumcheck/multilinear/sumcheck_prover.h"
#include "tachyon/crypto/sumcheck/multilinear/sumcheck_verifier.h"

namespace tachyon::crypto {

template <typename MLE>
class MultilinearSumcheck {
 public:
  template <size_t MaxDegree>
  [[nodiscard]] static bool RunInteractiveProtocol(
      const math::LinearCombination<MLE>& linear_combination,
      size_t num_variables) {
    using F = typename MLE::Field;

    const SumcheckProvingKey<MLE> proving_key =
        SumcheckProvingKey<MLE>::Build(linear_combination);
    const SumcheckVerifyingKey verifying_key =
        SumcheckVerifyingKey::Build(linear_combination);
    SumcheckProver<MLE, MaxDegree> prover(std::move(proving_key));
    SumcheckVerifier<MLE> verifier(std::move(verifying_key));
    SumcheckProverMsg<F, MaxDegree> prover_msg = prover.Round();
    SumcheckVerifierMsg<F> verifier_msg = verifier.Round(std::move(prover_msg));

    for (size_t i = 1; i < num_variables; ++i) {
      prover_msg = prover.Round(std::move(verifier_msg));
      verifier_msg = verifier.Round(std::move(prover_msg));
    }
    const F asserted_sum = linear_combination.Combine();
    Subclaim<F> subclaim;
    if (!verifier.CheckAndGenerateSubclaim(asserted_sum, &subclaim)) {
      return false;
    }
    F expected_sum = linear_combination.Evaluate(subclaim.point);

    return expected_sum == subclaim.expected_evaluation;
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_MULTILINEAR_SUMCHECK_H_
