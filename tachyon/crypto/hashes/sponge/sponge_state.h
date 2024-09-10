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
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon {
namespace crypto {

template <typename Params>
struct SpongeState {
  using F = typename Params::Field;
  // Current sponge's state (current elements in the permutation block)
  math::Vector<F> elements;

  // Current mode (whether its absorbing or squeezing)
  DuplexSpongeMode mode = DuplexSpongeMode::Absorbing();

  SpongeState() : elements(math::Vector<F>::Zero(Params::kWidth)) {}

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

template <typename Params>
class Copyable<crypto::SpongeState<Params>> {
 public:
  using F = typename Params::Field;
  static bool WriteTo(const crypto::SpongeState<Params>& state,
                      Buffer* buffer) {
    return buffer->WriteMany(state.elements, state.mode);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::SpongeState<Params>* state) {
    math::Vector<F> elements;
    crypto::DuplexSpongeMode mode;
    if (!buffer.ReadMany(&elements, &mode)) {
      return false;
    }

    state->elements = std::move(elements);
    state->mode = mode;
    return true;
  }

  static size_t EstimateSize(const crypto::SpongeState<Params>& state) {
    return base::EstimateSize(state.elements, state.mode);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_STATE_H_
