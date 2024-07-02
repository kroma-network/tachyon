// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_ZK_AIR_PLONKY3_CHALLENGER_DUPLEX_CHALLENGER_H_
#define TACHYON_ZK_AIR_PLONKY3_CHALLENGER_DUPLEX_CHALLENGER_H_

#include <utility>

#include "absl/container/inlined_vector.h"

#include "tachyon/crypto/hashes/sponge/sponge_state.h"
#include "tachyon/zk/air/plonky3/challenger/challenger.h"

namespace tachyon::zk::air::plonky3 {

template <typename Permutation, size_t W, size_t R>
class DuplexChallenger final
    : public Challenger<DuplexChallenger<Permutation, W, R>> {
 public:
  using F = typename Permutation::F;

  explicit DuplexChallenger(Permutation&& permutation)
      : permutation_(std::move(permutation)) {}

 private:
  friend class Challenger<DuplexChallenger<Permutation, W, R>>;

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
    for (size_t i = 0; i < W; ++i) {
      output_buffer_.push_back(state_[i]);
    }
  }

  crypto::SpongeState<F> state_{W};
  absl::InlinedVector<F, R> input_buffer_;
  absl::InlinedVector<F, W> output_buffer_;
  Permutation permutation_;
};

template <typename Permutation, size_t W, size_t R>
struct ChallengerTraits<DuplexChallenger<Permutation, W, R>> {
  using Field = typename Permutation::F;
};

}  // namespace tachyon::zk::air::plonky3

#endif  // TACHYON_ZK_AIR_PLONKY3_CHALLENGER_DUPLEX_CHALLENGER_H_
