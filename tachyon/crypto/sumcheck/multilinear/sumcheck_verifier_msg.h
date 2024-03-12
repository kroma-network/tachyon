// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_VERIFIER_MSG_H_
#define TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_VERIFIER_MSG_H_

#include <utility>

#include "tachyon/base/buffer/copyable.h"

namespace tachyon {
namespace crypto {

template <typename F>
struct SumcheckVerifierMsg {
  // Random value sampled by verifier.
  F random_value;

  bool operator==(const SumcheckVerifierMsg& other) const {
    return random_value == other.random_value;
  }
  bool operator!=(const SumcheckVerifierMsg& other) const {
    return !operator==(other);
  }
};

}  // namespace crypto

namespace base {

template <typename F>
class Copyable<crypto::SumcheckVerifierMsg<F>> {
 public:
  static bool WriteTo(const crypto::SumcheckVerifierMsg<F>& msg,
                      Buffer* buffer) {
    return buffer->Write(msg.random_value);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::SumcheckVerifierMsg<F>* msg) {
    F random_value;
    if (!buffer.Read(&random_value)) return false;
    *msg = {std::move(random_value)};
    return true;
  }

  static size_t EstimateSize(const crypto::SumcheckVerifierMsg<F>& msg) {
    return base::EstimateSize(msg.random_value);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_SUMCHECK_VERIFIER_MSG_H_
