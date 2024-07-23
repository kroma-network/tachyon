#ifndef TACHYON_ZK_PLONK_HALO2_PRIME_FIELD_CONVERSION_H_
#define TACHYON_ZK_PLONK_HALO2_PRIME_FIELD_CONVERSION_H_

#include <stdint.h>

#include "absl/numeric/int128.h"

#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/types/always_false.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon::zk::plonk::halo2 {

// See
// https://github.com/kroma-network/halo2curves/blob/c0ac193/src/derive/field.rs#L301-L303.
template <typename F>
static F FromUint128(absl::uint128 value) {
  if constexpr (F::N != 4) {
    static_assert(base::AlwaysFalse<F>);
  }
  uint64_t limbs[4] = {absl::Uint128Low64(value), absl::Uint128High64(value), 0,
                       0};
  return F(math::BigInt<4>(limbs));
}

// See
// https://github.com/kroma-network/halo2curves/blob/c0ac193/src/derive/field.rs#L29-L47.
template <typename F>
static F FromUint512(uint64_t limbs[8]) {
  if constexpr (F::N != 4) {
    static_assert(base::AlwaysFalse<F>);
  }
  F d0 = F::FromMontgomery(
      math::BigInt<4>({limbs[0], limbs[1], limbs[2], limbs[3]}));
  F d1 = F::FromMontgomery(
      math::BigInt<4>({limbs[4], limbs[5], limbs[6], limbs[7]}));
  // NOTE(chokobole): When performing d0 * F::Config::kMontgomeryR2 + d1 *
  // F::Config::kMontgomeryR3, the result may be incorrect. This is due to our
  // prime field multiplication, where we utilize unused modulus bits for
  // optimization purposes. However, the given |limbs| can sometimes exceed
  // the allowed scope of bits.
  math::BigInt<8> mul_result = d0.value().MulExtend(F::Config::kMontgomeryR2);
  math::BigInt<4> d2;
  math::BigInt<4>::MontgomeryReduce64<false>(mul_result, F::Config::kModulus,
                                             F::Config::kInverse64, &d2);
  math::BigInt<8> mul_result2 = d1.value().MulExtend(F::Config::kMontgomeryR3);
  math::BigInt<4> d3;
  math::BigInt<4>::MontgomeryReduce64<false>(mul_result2, F::Config::kModulus,
                                             F::Config::kInverse64, &d3);
  return F::FromMontgomery(d2) + F::FromMontgomery(d3);
}

template <typename F>
static F FromUint512(uint8_t bytes[64]) {
  base::ReadOnlyBuffer buffer(bytes, 64);
  uint64_t limbs[8];
  for (size_t i = 0; i < 8; ++i) {
    CHECK(buffer.Read64LE(&limbs[i]));
  }
  return FromUint512<F>(limbs);
}

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PRIME_FIELD_CONVERSION_H_
