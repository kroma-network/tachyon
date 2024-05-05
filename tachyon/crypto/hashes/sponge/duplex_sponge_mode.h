// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_DUPLEX_SPONGE_MODE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_DUPLEX_SPONGE_MODE_H_

#include <stddef.h>

#include "tachyon/base/buffer/copyable.h"

namespace tachyon {
namespace crypto {

// The mode structure for duplex sponge.
struct TACHYON_EXPORT DuplexSpongeMode {
  enum class Type {
    // The sponge is currently absorbing data.
    kAbsorbing,
    // The sponge is currently squeezing data out.
    kSqueezing,
  };

  constexpr DuplexSpongeMode() = default;

  constexpr static DuplexSpongeMode Absorbing(size_t next_index = 0) {
    return {Type::kAbsorbing, next_index};
  }
  constexpr static DuplexSpongeMode Squeezing(size_t next_index = 0) {
    return {Type::kSqueezing, next_index};
  }

  Type type = Type::kAbsorbing;
  // When |type| is |kAbsorbing|, it is interpreted as next position of the
  // state to be XOR-ed when absorbing.
  // When |type| is |kSqueezing|, it is interpreted as next position of the
  // state to be outputted when squeezing.
  size_t next_index = 0;

  bool operator==(const DuplexSpongeMode& other) const {
    return type == other.type && next_index == other.next_index;
  }
  bool operator!=(const DuplexSpongeMode& other) const {
    return !operator==(other);
  }

 private:
  friend class base::Copyable<DuplexSpongeMode>;

  constexpr DuplexSpongeMode(Type type, size_t next_index)
      : type(type), next_index(next_index) {}
};

}  // namespace crypto

namespace base {

template <>
class Copyable<crypto::DuplexSpongeMode> {
 public:
  static bool WriteTo(const crypto::DuplexSpongeMode& mode, Buffer* buffer) {
    return buffer->WriteMany(mode.type, mode.next_index);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::DuplexSpongeMode* mode) {
    crypto::DuplexSpongeMode::Type type;
    size_t next_index;
    if (!buffer.ReadMany(&type, &next_index)) {
      return false;
    }

    *mode = {type, next_index};
    return true;
  }

  static size_t EstimateSize(const crypto::DuplexSpongeMode& mode) {
    return base::EstimateSize(mode.type, mode.next_index);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_DUPLEX_SPONGE_MODE_H_
