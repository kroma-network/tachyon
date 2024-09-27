#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PARAMS_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PARAMS_H_

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "tachyon/base/types/always_false.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_vendor.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

namespace tachyon::crypto {

template <typename F, size_t Width, uint64_t Alpha>
constexpr size_t GetPoseidon2PartialRounds() {
  using PrimeField = math::MaybeUnpack<F>;

  constexpr auto SelectRounds = [](uint64_t alpha,
                                   const std::array<size_t, 5>& rounds) {
    switch (alpha) {
      case 3:
        return rounds[0];
      case 5:
        return rounds[1];
      case 7:
        return rounds[2];
      case 9:
        return rounds[3];
      case 11:
        return rounds[4];
      default:
        NOTREACHED();
        return size_t{0};
    }
  };

  if constexpr (PrimeField::Config::kModulusBits == 31) {
    if constexpr (Width == 16) {
      return SelectRounds(Alpha, std::array<size_t, 5>{20, 14, 13, 13, 13});
    } else if constexpr (Width == 24) {
      return SelectRounds(Alpha, std::array<size_t, 5>{23, 22, 21, 21, 21});
    } else {
      static_assert(base::AlwaysFalse<F>);
    }
  } else if constexpr (PrimeField::Config::kModulusBits == 64) {
    if constexpr (Width == 8) {
      return SelectRounds(Alpha, std::array<size_t, 5>{41, 27, 22, 19, 17});
    } else if constexpr (Width == 16) {
      return SelectRounds(Alpha, std::array<size_t, 5>{42, 27, 22, 20, 18});
    } else if constexpr (Width == 24) {
      return SelectRounds(Alpha, std::array<size_t, 5>{47, 27, 22, 20, 18});
    } else {
      static_assert(base::AlwaysFalse<F>);
    }
  }
  return size_t{56};
}

template <Poseidon2Vendor ExternalMatrixVendor,
          Poseidon2Vendor InternalMatrixVendor, typename _Field, size_t Rate,
          uint32_t Alpha, size_t Capacity = 1, size_t FullRounds = 8,
          size_t PartialRounds =
              (GetPoseidon2PartialRounds<_Field, Rate + Capacity, Alpha>())>
struct Poseidon2Params {
  using Field = _Field;

  constexpr static Poseidon2Vendor kExternalMatrixVendor = ExternalMatrixVendor;
  constexpr static Poseidon2Vendor kInternalMatrixVendor = InternalMatrixVendor;

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

template <typename F, size_t Rate, uint32_t Alpha>
struct Poseidon2ParamsTraits;

template <typename Params>
auto GetPoseidon2InternalDiagonalArray() {
  using PrimeField = math::MaybeUnpack<typename Params::Field>;
  return Poseidon2ParamsTraits<PrimeField, Params::kRate, Params::kAlpha>::
      GetPoseidon2InternalDiagonalArray();
}

template <typename Params>
auto GetPoseidon2InternalShiftArray() {
  using PrimeField = math::MaybeUnpack<typename Params::Field>;
  return Poseidon2ParamsTraits<PrimeField, Params::kRate, Params::kAlpha>::
      GetPoseidon2InternalShiftArray();
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PARAMS_H_
