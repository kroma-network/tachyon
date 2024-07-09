// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_STATE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_STATE_H_

#include <sstream>
#include <string>
#include <utility>

#include "tachyon/base/strings/string_util.h"
#include "tachyon/crypto/hashes/sponge/duplex_sponge_mode.h"
#include "tachyon/crypto/hashes/sponge/sponge_config.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon {
namespace crypto {

template <typename F>
struct SpongeState {
  // Current sponge's state (current elements in the permutation block)
  math::Vector<F> elements;

  // Current mode (whether its absorbing or squeezing)
  DuplexSpongeMode mode = DuplexSpongeMode::Absorbing();

  SpongeState() = default;
  explicit SpongeState(const SpongeConfig& config)
      : SpongeState(config.rate + config.capacity) {}
  explicit SpongeState(size_t size) : elements(size) {
    for (size_t i = 0; i < size; ++i) {
      elements[i] = F::Zero();
    }
  }

  size_t size() const { return elements.size(); }

  F& operator[](size_t idx) { return elements[idx]; }
  const F& operator[](size_t idx) const { return elements[idx]; }

  bool operator==(const SpongeState& other) const {
    return elements == other.elements && mode == other.mode;
  }
  bool operator!=(const SpongeState& other) const { return !operator==(other); }

  std::string ToString() const { return base::ContainerToString(elements); }

  std::string ToHexString(bool pad_zero = false) const {
    return base::ContainerToHexString(elements, pad_zero);
  }
};

}  // namespace crypto

namespace base {

template <typename F>
class Copyable<crypto::SpongeState<F>> {
 public:
  static bool WriteTo(const crypto::SpongeState<F>& state, Buffer* buffer) {
    return buffer->WriteMany(state.elements, state.mode);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::SpongeState<F>* state) {
    math::Vector<F> elements;
    crypto::DuplexSpongeMode mode;
    if (!buffer.ReadMany(&elements, &mode)) {
      return false;
    }

    state->elements = std::move(elements);
    state->mode = mode;
    return true;
  }

  static size_t EstimateSize(const crypto::SpongeState<F>& state) {
    return base::EstimateSize(state.elements, state.mode);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_STATE_H_
