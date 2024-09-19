// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_CHALLENGER_DUPLEX_CHALLENGER_H_
#define TACHYON_CRYPTO_CHALLENGER_DUPLEX_CHALLENGER_H_

#include <utility>

#include "absl/container/inlined_vector.h"

#include "tachyon/crypto/challenger/challenger.h"
#include "tachyon/crypto/hashes/sponge/sponge_state.h"

namespace tachyon {

class HintableTest_DuplexChallenger_Test;

namespace crypto {

template <typename Permutation, size_t R>
class DuplexChallenger final
    : public Challenger<DuplexChallenger<Permutation, R>> {
 public:
  using Params = typename Permutation::Params;
  using F = typename Params::Field;

  DuplexChallenger() = default;
  explicit DuplexChallenger(Permutation&& permutation)
      : permutation_(std::move(permutation)) {}

  const SpongeState<Params>& state() const { return state_; }
  const absl::InlinedVector<F, R>& input_buffer() const {
    return input_buffer_;
  }
  const absl::InlinedVector<F, Params::kWidth>& output_buffer() const {
    return output_buffer_;
  }

 private:
  friend class Challenger<DuplexChallenger<Permutation, R>>;
  friend class tachyon::HintableTest_DuplexChallenger_Test;

  // Challenger methods
  void DoObserve(const F& value) {
    output_buffer_.clear();

    input_buffer_.push_back(value);

    if (input_buffer_.size() == R) {
      Duplex();
    }
  }

  F DoSample() {
    if (!input_buffer_.empty() || output_buffer_.empty()) {
      Duplex();
    }

    F ret = std::move(output_buffer_.back());
    output_buffer_.pop_back();
    return ret;
  }

  void Duplex() {
    for (size_t i = 0; i < input_buffer_.size(); ++i) {
      state_[i] = std::move(input_buffer_[i]);
    }
    input_buffer_.clear();

    permutation_.Permute(state_);

    output_buffer_.clear();
    for (size_t i = 0; i < Params::kWidth; ++i) {
      output_buffer_.push_back(state_[i]);
    }
  }

  SpongeState<Params> state_;
  absl::InlinedVector<F, R> input_buffer_;
  absl::InlinedVector<F, Params::kWidth> output_buffer_;
  Permutation permutation_;
};

template <typename Permutation, size_t R>
struct ChallengerTraits<DuplexChallenger<Permutation, R>> {
  using Field = typename Permutation::F;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_CHALLENGER_DUPLEX_CHALLENGER_H_
