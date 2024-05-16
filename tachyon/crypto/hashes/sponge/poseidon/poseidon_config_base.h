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

template <typename PrimeField>
struct PoseidonConfigBase {
  using F = PrimeField;

  // Number of rounds in a full-round operation.
  size_t full_rounds = 0;

  // Number of rounds in a partial-round operation.
  size_t partial_rounds = 0;

  // Exponent used in S-boxes.
  uint64_t alpha = 0;

  // Additive Round Keys added before each MDS matrix application to make it an
  // affine shift. They are indexed by |ark[round_num][state_element_index]|.
  math::Matrix<PrimeField> ark;

  // The rate (in terms of number of field elements).
  // See https://iacr.org/archive/eurocrypt2008/49650180/49650180.pdf
  size_t rate = 0;

  // The capacity (in terms of number of field elements).
  size_t capacity = 0;

  PoseidonConfigBase() = default;
  PoseidonConfigBase(size_t full_rounds, size_t partial_rounds, uint64_t alpha,
                     const math::Matrix<PrimeField>& ark, size_t rate,
                     size_t capacity)
      : full_rounds(full_rounds),
        partial_rounds(partial_rounds),
        alpha(alpha),
        ark(ark),
        rate(rate),
        capacity(capacity) {}
  PoseidonConfigBase(size_t full_rounds, size_t partial_rounds, uint64_t alpha,
                     math::Matrix<PrimeField>&& ark, size_t rate,
                     size_t capacity)
      : full_rounds(full_rounds),
        partial_rounds(partial_rounds),
        alpha(alpha),
        ark(std::move(ark)),
        rate(rate),
        capacity(capacity) {}

  virtual ~PoseidonConfigBase() = default;

  virtual bool IsValid() const {
    return static_cast<size_t>(ark.rows()) == full_rounds + partial_rounds &&
           static_cast<size_t>(ark.cols()) == rate + capacity;
  }

  bool operator==(const PoseidonConfigBase& other) const {
    return full_rounds == other.full_rounds &&
           partial_rounds == other.partial_rounds && alpha == other.alpha &&
           ark == other.ark && rate == other.rate && capacity == other.capacity;
  }
  bool operator!=(const PoseidonConfigBase& other) const {
    return !operator==(other);
  }
};

}  // namespace crypto

namespace base {

template <typename PrimeField>
class Copyable<crypto::PoseidonConfigBase<PrimeField>> {
 public:
  static bool WriteTo(const crypto::PoseidonConfigBase<PrimeField>& config,
                      Buffer* buffer) {
    return buffer->WriteMany(config.full_rounds, config.partial_rounds,
                             config.alpha, config.ark, config.rate,
                             config.capacity);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonConfigBase<PrimeField>* config) {
    size_t full_rounds;
    size_t partial_rounds;
    uint64_t alpha;
    math::Matrix<PrimeField> ark;
    size_t rate;
    size_t capacity;
    if (!buffer.ReadMany(&full_rounds, &partial_rounds, &alpha, &ark, &rate,
                         &capacity)) {
      return false;
    }

    *config = {full_rounds,    partial_rounds, alpha,
               std::move(ark), rate,           capacity};
    return true;
  }

  static size_t EstimateSize(
      const crypto::PoseidonConfigBase<PrimeField>& config) {
    return base::EstimateSize(config.full_rounds, config.partial_rounds,
                              config.alpha, config.ark, config.rate,
                              config.capacity);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_BASE_H_
