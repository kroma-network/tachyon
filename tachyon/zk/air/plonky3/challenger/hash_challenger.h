// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_ZK_AIR_PLONKY3_CHALLENGER_HASH_CHALLENGER_H_
#define TACHYON_ZK_AIR_PLONKY3_CHALLENGER_HASH_CHALLENGER_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/air/plonky3/challenger/challenger.h"

namespace tachyon::zk::air::plonky3 {

template <typename Hasher>
class HashChallenger final : public Challenger<HashChallenger<Hasher>> {
 public:
  using F = typename Hasher::F;

  explicit HashChallenger(Hasher&& hasher) : hasher_(std::move(hasher)) {}
  HashChallenger(const std::vector<F>& input_buffer, Hasher&& hasher)
      : input_buffer_(input_buffer), hasher_(std::move(hasher)) {}
  HashChallenger(std::vector<F>&& input_buffer, Hasher&& hasher)
      : input_buffer_(std::move(input_buffer)), hasher_(std::move(hasher)) {}

 private:
  friend class Challenger<HashChallenger<Hasher>>;

  // Challenger methods
  void DoObserve(const F& value) {
    output_buffer_.clear();

    input_buffer_.push_back(value);
  }

  F DoSample() {
    if (output_buffer_.empty()) {
      Flush();
    }

    F ret = std::move(output_buffer_.back());
    output_buffer_.pop_back();
    return ret;
  }

  void Flush() {
    auto output = hasher_.Hash(input_buffer_);

    output_buffer_ =
        base::Map(output, [](F& value) { return std::move(value); });
    input_buffer_ = output_buffer_;
  }

  std::vector<F> input_buffer_;
  std::vector<F> output_buffer_;
  Hasher hasher_;
};

template <typename Hasher>
struct ChallengerTraits<HashChallenger<Hasher>> {
  using Field = typename Hasher::F;
};

}  // namespace tachyon::zk::air::plonky3

#endif  // TACHYON_ZK_AIR_PLONKY3_CHALLENGER_HASH_CHALLENGER_H_
