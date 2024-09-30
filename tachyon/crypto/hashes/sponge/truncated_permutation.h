// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_TRUNCATED_PERMUTATION_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_TRUNCATED_PERMUTATION_H_

#include <stddef.h>

#include <array>
#include <utility>

#include "tachyon/crypto/hashes/compressor.h"
#include "tachyon/crypto/hashes/sponge/sponge.h"

namespace tachyon::crypto {

template <typename Derived, size_t Chunk, size_t N>
class TruncatedPermutation final
    : public Compressor<TruncatedPermutation<Derived, Chunk, N>> {
 public:
  constexpr static size_t kChunk = Chunk;
  constexpr static size_t kN = N;

  using Params = typename Derived::Params;
  using F = typename CryptographicSpongeTraits<Derived>::F;

  TruncatedPermutation() = default;
  explicit TruncatedPermutation(const Derived& derived) : derived_(derived) {}
  explicit TruncatedPermutation(Derived&& derived)
      : derived_(std::move(derived)) {}

  const Derived& derived() const { return derived_; }

 private:
  friend class Compressor<TruncatedPermutation<Derived, Chunk, N>>;

  SpongeState<Params> CreateEmptyState() const { return SpongeState<Params>(); }

  template <typename T>
  std::array<F, Chunk> DoCompress(SpongeState<Params>& state,
                                  const T& input) const {
    size_t idx = 0;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < Chunk; ++j) {
        state[idx++] = input[i][j];
      }
    }
    derived_.Permute(state);
    std::array<F, Chunk> ret;
    for (size_t i = 0; i < Chunk; ++i) {
      ret[i] = std::move(state[i]);
    }
    return ret;
  }

  Derived derived_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_TRUNCATED_PERMUTATION_H_
