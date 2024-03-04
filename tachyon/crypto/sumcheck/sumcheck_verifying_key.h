// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_SUMCHECK_SUMCHECK_VERIFYING_KEY_H_
#define TACHYON_CRYPTO_SUMCHECK_SUMCHECK_VERIFYING_KEY_H_

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/random.h"
#include "tachyon/math/polynomials/multivariate/linear_combination.h"

namespace tachyon {
namespace crypto {

struct TACHYON_EXPORT VerifyingKey {
  size_t max_evaluations;
  size_t num_variables;

  template <typename MLE>
  static VerifyingKey Build(
      const math::LinearCombination<MLE>& linear_combination) {
    return {linear_combination.max_evaluations(),
            linear_combination.num_variables()};
  }

  bool operator==(const VerifyingKey& other) const {
    return max_evaluations == other.max_evaluations &&
           num_variables == other.num_variables;
  }
  bool operator!=(const VerifyingKey& other) const {
    return !operator==(other);
  }

  static VerifyingKey Random() {
    return {base::Uniform(base::Range<size_t>()),
            base::Uniform(base::Range<size_t>())};
  }
};

}  // namespace crypto

namespace base {

template <>
class Copyable<crypto::VerifyingKey> {
 public:
  static bool WriteTo(const crypto::VerifyingKey& verifying_key,
                      Buffer* buffer) {
    return buffer->WriteMany(verifying_key.max_evaluations,
                             verifying_key.num_variables);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::VerifyingKey* verifying_key) {
    size_t max_evaluations;
    size_t num_variables;
    if (!buffer.ReadMany(&max_evaluations, &num_variables)) return false;
    *verifying_key = {max_evaluations, num_variables};
    return true;
  }

  static size_t EstimateSize(const crypto::VerifyingKey& verifying_key) {
    return base::EstimateSize(verifying_key.max_evaluations,
                              verifying_key.num_variables);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_SUMCHECK_SUMCHECK_VERIFYING_KEY_H_
