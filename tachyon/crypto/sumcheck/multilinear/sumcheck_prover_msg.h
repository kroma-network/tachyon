// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_PROVER_MSG_H_
#define TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_PROVER_MSG_H_

#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon {
namespace crypto {

template <typename F, size_t MaxDegree>
struct SumcheckProverMsg {
  // Evaluations on P(0), P(1), P(2), ...
  math::UnivariateEvaluations<F, MaxDegree> evaluations;

  bool operator==(const SumcheckProverMsg& other) const {
    return evaluations == other.evaluations;
  }
  bool operator!=(const SumcheckProverMsg& other) const {
    return !operator==(other);
  }
};

}  // namespace crypto

namespace base {

template <typename F, size_t MaxDegree>
class Copyable<crypto::SumcheckProverMsg<F, MaxDegree>> {
 public:
  static bool WriteTo(const crypto::SumcheckProverMsg<F, MaxDegree>& msg,
                      Buffer* buffer) {
    return buffer->Write(msg.evaluations);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::SumcheckProverMsg<F, MaxDegree>* msg) {
    math::UnivariateEvaluations<F, MaxDegree> evaluations;
    if (!buffer.Read(&evaluations)) return false;
    *msg = {std::move(evaluations)};
    return true;
  }

  static size_t EstimateSize(
      const crypto::SumcheckProverMsg<F, MaxDegree>& msg) {
    return base::EstimateSize(msg.evaluations);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_PROVER_MSG_H_
