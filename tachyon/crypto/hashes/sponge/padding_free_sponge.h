// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_PADDING_FREE_SPONGE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_PADDING_FREE_SPONGE_H_

#include <stddef.h>

#include <array>
#include <utility>

#include "tachyon/crypto/hashes/hasher.h"
#include "tachyon/crypto/hashes/sponge/sponge.h"

namespace tachyon::crypto {

template <typename Derived, size_t Rate, size_t Out>
class PaddingFreeSponge final
    : public Hasher<PaddingFreeSponge<Derived, Rate, Out>> {
 public:
  using F = typename CryptographicSpongeTraits<Derived>::F;

  PaddingFreeSponge() = default;
  explicit PaddingFreeSponge(const Derived& derived) : derived_(derived) {}
  explicit PaddingFreeSponge(Derived&& derived)
      : derived_(std::move(derived)) {}

 private:
  friend class Hasher<PaddingFreeSponge<Derived, Rate, Out>>;

  SpongeState<F> CreateEmptyState() const {
    return SpongeState<F>(derived_.config);
  }

  template <typename T>
  std::array<F, Out> DoHash(SpongeState<F>& state, const T& input) const {
    LOG(ERROR) << "std::size(input): " << std::size(input);
    LOG(ERROR) << "Rate: " << Rate;
    LOG(ERROR) << "Input";
    for (size_t i = 0; i < std::size(input); i += Rate) {
      for (size_t j = 0; j < Rate; ++j) {
        if (i + j < std::size(input)) {
          state[j] = input[i + j];
        }
        LOG(ERROR) << "state[" << j << "]: " << state[j];
      }
      derived_.Permute(state);
    }
    LOG(ERROR) << "Result";
    std::array<F, Out> ret;
    for (size_t i = 0; i < Out; ++i) {
      LOG(ERROR) << "state[" << i << "]: " << state[i];
      ret[i] = std::move(state[i]);
    }
    return ret;
  }

  Derived derived_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_PADDING_FREE_SPONGE_H_
