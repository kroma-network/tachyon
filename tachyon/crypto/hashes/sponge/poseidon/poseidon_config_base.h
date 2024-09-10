// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_BASE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_BASE_H_

#include <stddef.h>
#include <stdint.h>

#include <utility>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon {
namespace crypto {

template <typename Params>
struct PoseidonConfigBase {
  using F = typename Params::Field;

  // Additive Round Keys added before each MDS matrix application to make it an
  // affine shift. They are indexed by |ark[round_num][state_element_index]|.
  math::Matrix<F> ark;

  PoseidonConfigBase() = default;
  explicit PoseidonConfigBase(const math::Matrix<F>& ark) : ark(ark) {}
  explicit PoseidonConfigBase(math::Matrix<F>&& ark) : ark(std::move(ark)) {}

  virtual ~PoseidonConfigBase() = default;

  virtual bool IsValid() const {
    return static_cast<size_t>(ark.rows()) ==
               Params::kFullRounds + Params::kPartialRounds &&
           static_cast<size_t>(ark.cols()) == Params::kRate + Params::kCapacity;
  }

  bool operator==(const PoseidonConfigBase& other) const {
    return ark == other.ark;
  }
  bool operator!=(const PoseidonConfigBase& other) const {
    return !operator==(other);
  }
};

}  // namespace crypto

namespace base {

template <typename Params>
class Copyable<crypto::PoseidonConfigBase<Params>> {
 public:
  static bool WriteTo(const crypto::PoseidonConfigBase<Params>& config,
                      Buffer* buffer) {
    return buffer->Write(config.ark);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonConfigBase<Params>* config) {
    math::Matrix<typename Params::Field> ark;
    if (!buffer.Read(&ark)) {
      return false;
    }

    config->ark = std::move(ark);
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonConfigBase<Params>& config) {
    return base::EstimateSize(config.ark);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_BASE_H_
