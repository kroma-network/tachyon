#ifndef VENDORS_CIRCOM_CIRCOMLIB_BASE_PRIME_FIELD_H_
#define VENDORS_CIRCOM_CIRCOMLIB_BASE_PRIME_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon::circom {

struct PrimeField {
  std::vector<uint8_t> bytes;

  bool operator==(const PrimeField& other) const {
    return bytes == other.bytes;
  }
  bool operator!=(const PrimeField& other) const {
    return bytes != other.bytes;
  }

  template <size_t N>
  math::BigInt<N> ToBigInt() const {
    CHECK_EQ(bytes.size() / 8, N);
    return math::BigInt<N>::FromBytesLE(bytes);
  }

  template <size_t N>
  static PrimeField FromBigInt(const math::BigInt<N>& big_int) {
    std::array<uint8_t, N * 8> bytes = big_int.ToBytesLE();
    return {{bytes.begin(), bytes.end()}};
  }

  template <typename F>
  F ToNative() const {
    return F::FromMontgomery(ToBigInt<F::kLimbNums>());
  }

  template <typename F>
  static PrimeField FromNative(const F& prime_field) {
    return FromBigInt(prime_field.ToMontgomery());
  }

  bool Read(const base::ReadOnlyBuffer& buffer, uint32_t num_bytes = 0) {
    base::EndianAutoReset reset(buffer, base::Endian::kLittle);
    if (num_bytes == 0) {
      if (!buffer.Read(&num_bytes)) return false;
      if (num_bytes % 8 != 0) {
        LOG(ERROR) << "field size is not a multiple of 8";
        return false;
      }
    }
    bytes.resize(num_bytes);
    return buffer.Read(bytes.data(), bytes.size());
  }

  std::string ToString() const;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_BASE_PRIME_FIELD_H_
