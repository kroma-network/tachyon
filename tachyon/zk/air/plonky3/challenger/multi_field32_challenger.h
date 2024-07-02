// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_ZK_AIR_PLONKY3_CHALLENGER_MULTI_FIELD32_CHALLENGER_H_
#define TACHYON_ZK_AIR_PLONKY3_CHALLENGER_MULTI_FIELD32_CHALLENGER_H_

#include <algorithm>
#include <utility>

#include "absl/container/inlined_vector.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/hashes/sponge/sponge_state.h"
#include "tachyon/zk/air/plonky3/base/multi_field32_conversions.h"
#include "tachyon/zk/air/plonky3/challenger/challenger.h"

namespace tachyon::zk::air::plonky3 {

// A challenger that operates natively on |BigF| but produces challenges of
// |SmallF|.
//
// Used for optimizing the cost of recursive proof verification of STARKs in
// SNARKs.
//
// SAFETY: There are some bias complications with using this challenger. In
// particular, samples are actually random in [0, 2⁶⁴) and then reduced to be
// in |SmallF|.
template <typename SmallF, typename Permutation, size_t W>
class MultiField32Challenger final
    : public Challenger<MultiField32Challenger<SmallF, Permutation, W>> {
 public:
  using BigF = typename Permutation::F;

  static_assert(BigF::Config::kModulusBits > 64);
  static_assert(SmallF::Config::kModulusBits <= 32);

  constexpr static size_t kNumFElements = BigF::Config::kModulusBits / 64;
  constexpr static size_t R = kNumFElements * W;

  explicit MultiField32Challenger(Permutation&& permutation)
      : permutation_(std::move(permutation)) {}

 private:
  friend class Challenger<MultiField32Challenger<SmallF, Permutation, W>>;

  // Challenger methods
  void DoObserve(const SmallF& value) {
    output_buffer_.clear();

    input_buffer_.push_back(value);

    if (input_buffer_.size() == R) {
      Duplex();
    }
  }

  SmallF DoSample() {
    if (!input_buffer_.empty() || output_buffer_.empty()) {
      Duplex();
    }

    SmallF ret = std::move(output_buffer_.back());
    output_buffer_.pop_back();
    return ret;
  }

  void Duplex() {
    for (size_t i = 0;
         i < (input_buffer_.size() + kNumFElements - 1) / kNumFElements; ++i) {
      size_t start = i * kNumFElements;
      size_t len = std::min(input_buffer_.size() - start, kNumFElements);
      state_[i] =
          Reduce<BigF>(absl::Span<const SmallF>(&input_buffer_[start], len));
    }
    input_buffer_.clear();

    permutation_.Permute(state_);

    output_buffer_.clear();
    for (size_t i = 0; i < W; ++i) {
      std::array<SmallF, kNumFElements> values = Split<SmallF>(state_[i]);
      for (size_t j = 0; j < kNumFElements; ++j) {
        output_buffer_.push_back(values[j]);
      }
    }
  }

  crypto::SpongeState<BigF> state_{W};
  absl::InlinedVector<SmallF, R> input_buffer_;
  absl::InlinedVector<SmallF, W * kNumFElements> output_buffer_;
  Permutation permutation_;
};

template <typename SmallF, typename Permutation, size_t W>
struct ChallengerTraits<MultiField32Challenger<SmallF, Permutation, W>> {
  using Field = SmallF;
};

}  // namespace tachyon::zk::air::plonky3

#endif  // TACHYON_ZK_AIR_PLONKY3_CHALLENGER_MULTI_FIELD32_CHALLENGER_H_
