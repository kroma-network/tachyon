#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_PARAMS_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_PARAMS_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/baby_bear/internal/baby_bear.h"

namespace tachyon::crypto {

template <typename _Field, size_t Rate, uint32_t Alpha, size_t FullRounds,
          size_t PartialRounds, size_t Capacity = 1>
struct PoseidonParams {
  using Field = _Field;

  // The rate (in terms of number of field elements).
  // See https://iacr.org/archive/eurocrypt2008/49650180/49650180.pdf
  constexpr static size_t kRate = Rate;
  // The capacity (in terms of number of field elements).
  constexpr static size_t kCapacity = Capacity;
  constexpr static size_t kWidth = Rate + Capacity;
  // NOTE(ashjeong): |Alpha| is also referred to as |D|
  // Exponent used in S-boxes.
  constexpr static uint32_t kAlpha = Alpha;
  // Number of rounds in a full-round operation.
  constexpr static size_t kFullRounds = FullRounds;
  // Number of rounds in a partial-round operation.
  constexpr static size_t kPartialRounds = PartialRounds;
};

// NOTE(ashjeong): The variables names' ending number refers to the |Width|;
// however, note that the |PartialRounds| and |FullRounds| depend on both
// |Width| and |Alpha|.
using BabyBearPoseidonParams16 = PoseidonParams<math::BabyBear, 15, 7, 8, 22>;
using BabyBearPoseidonParams24 = PoseidonParams<math::BabyBear, 23, 7, 8, 22>;
using BN254PoseidonParams9 = PoseidonParams<math::bn254::Fr, 8, 5, 8, 63>;
using BN254PoseidonParams5 = PoseidonParams<math::bn254::Fr, 4, 5, 8, 60>;

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_PARAMS_H_
