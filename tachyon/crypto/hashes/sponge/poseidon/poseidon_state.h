// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_STATE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_STATE_H_

#include <sstream>
#include <string>
#include <utility>

#include "tachyon/crypto/hashes/sponge/sponge.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon {
namespace crypto {

template <typename PrimeField>
struct PoseidonState {
  // Current sponge's state (current elements in the permutation block)
  math::Vector<PrimeField> elements;

  // Current mode (whether its absorbing or squeezing)
  DuplexSpongeMode mode = DuplexSpongeMode::Absorbing();

  PoseidonState() = default;
  explicit PoseidonState(size_t size) : elements(size) {
    for (size_t i = 0; i < size; ++i) {
      elements[i] = PrimeField::Zero();
    }
  }

  size_t size() const { return elements.size(); }

  PrimeField& operator[](size_t idx) { return elements[idx]; }
  const PrimeField& operator[](size_t idx) const { return elements[idx]; }

  bool operator==(const PoseidonState& other) const {
    return elements == other.elements && mode == other.mode;
  }
  bool operator!=(const PoseidonState& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "[";
    for (Eigen::Index i = 0; i < elements.size(); ++i) {
      ss << elements[i].ToString();
      if (i != elements.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    return ss.str();
  }

  std::string ToHexString(bool pad_zero = false) const {
    std::stringstream ss;
    ss << "[";
    for (Eigen::Index i = 0; i < elements.size(); ++i) {
      ss << elements[i].ToHexString(pad_zero);
      if (i != elements.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    return ss.str();
  }
};

}  // namespace crypto

namespace base {

template <typename PrimeField>
class Copyable<crypto::PoseidonState<PrimeField>> {
 public:
  static bool WriteTo(const crypto::PoseidonState<PrimeField>& state,
                      Buffer* buffer) {
    return buffer->WriteMany(state.elements, state.mode);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonState<PrimeField>* state) {
    math::Vector<PrimeField> elements;
    crypto::DuplexSpongeMode mode;
    if (!buffer.ReadMany(&elements, &mode)) {
      return false;
    }

    state->elements = std::move(elements);
    state->mode = mode;
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonState<PrimeField>& state) {
    return base::EstimateSize(state.elements, state.mode);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_STATE_H_
