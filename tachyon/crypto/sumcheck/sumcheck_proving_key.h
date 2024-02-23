#ifndef TACHYON_CRYPTO_SUMCHECK_SUMCHECK_PROVING_KEY_H_
#define TACHYON_CRYPTO_SUMCHECK_SUMCHECK_PROVING_KEY_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/sumcheck/sumcheck_verifying_key.h"

namespace tachyon {
namespace crypto {

template <typename MLE>
struct ProvingKey {
  VerifyingKey verifying_key;
  std::vector<math::LinearCombinationTerm<typename MLE::Field>> terms;
  std::vector<MLE> flattened_ml_evaluations;

  static ProvingKey Build(
      const math::LinearCombination<MLE>& linear_combination) {
    std::vector<MLE> flattened_ml_evaluations =
        base::Map(linear_combination.flattened_ml_evaluations(),
                  [](const std::shared_ptr<MLE> ptr) { return *ptr.get(); });
    return {VerifyingKey::Build(linear_combination), linear_combination.terms(),
            std::move(flattened_ml_evaluations)};
  }

  bool operator==(const ProvingKey& other) const {
    return verifying_key == other.verifying_key && terms == other.terms &&
           flattened_ml_evaluations == flattened_ml_evaluations;
  }
  bool operator!=(const ProvingKey& other) const { return !operator==(other); }
};

}  // namespace crypto

namespace base {

template <typename MLE>
class Copyable<crypto::ProvingKey<MLE>> {
 public:
  static bool WriteTo(const crypto::ProvingKey<MLE>& proving_key,
                      Buffer* buffer) {
    return buffer->WriteMany(proving_key.verifying_key, proving_key.terms,
                             proving_key.flattened_ml_evaluations);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::ProvingKey<MLE>* proving_key) {
    crypto::VerifyingKey verifying_key;
    std::vector<math::LinearCombinationTerm<typename MLE::Field>> terms;
    std::vector<MLE> flattened_ml_evaluations;
    if (!buffer.ReadMany(&verifying_key, &terms, &flattened_ml_evaluations))
      return false;
    *proving_key = {std::move(verifying_key), std::move(terms),
                    std::move(flattened_ml_evaluations)};
    return true;
  }

  static size_t EstimateSize(const crypto::ProvingKey<MLE>& proving_key) {
    return base::EstimateSize(proving_key.verifying_key, proving_key.terms,
                              proving_key.flattened_ml_evaluations);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_SUMCHECK_SUMCHECK_PROVING_KEY_H_
