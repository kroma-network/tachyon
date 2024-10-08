// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_H_

#include <array>
#include <utility>

#include "absl/types/span.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config_base.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_config_entry.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"
#include "tachyon/math/matrix/matrix_utils.h"

namespace tachyon {
namespace crypto {

// ARK(AddRoundKey) is a matrix that contains an ARC(AddRoundConstant) array in
// each row. Each constant is added to the |state| of each round of Poseidon.
// TODO(chokobole): add comment
template <typename F>
void FindPoseidon2ARK(const PoseidonGrainLFSRConfig& config,
                      math::Matrix<F>& ark) {
  PoseidonGrainLFSR<F> lfsr(config);
  ark = math::Matrix<F>(config.num_full_rounds + config.num_partial_rounds,
                        config.state_len);
  size_t partial_rounds_start = config.num_full_rounds / 2;
  size_t partial_rounds_end =
      config.num_full_rounds / 2 + config.num_partial_rounds;
  for (size_t i = 0; i < config.num_full_rounds + config.num_partial_rounds;
       ++i) {
    if (i < partial_rounds_start || i >= partial_rounds_end) {
      ark.row(i) = lfsr.GetFieldElementsRejectionSampling(config.state_len);
    } else {
      ark.row(i)[0] = lfsr.GetFieldElementsRejectionSampling(1)[0];
    }
  }
  // TODO(chokobole): Enable generating |internal_diagonal_mins_one|.
}

template <typename Params>
struct Poseidon2Config : public PoseidonConfigBase<Params> {
  using F = typename Params::Field;
  using PrimeField = math::MaybeUnpack<F>;

  math::Vector<F> internal_diagonal_minus_one;
  math::Vector<uint8_t> internal_shifts;

  Poseidon2Config() = default;
  Poseidon2Config(const PoseidonConfigBase<Params>& base,
                  const math::Vector<F>& internal_diagonal_minus_one)
      : PoseidonConfigBase<Params>(base),
        internal_diagonal_minus_one(internal_diagonal_minus_one) {}
  Poseidon2Config(PoseidonConfigBase<Params>&& base,
                  math::Vector<F>&& internal_diagonal_minus_one)
      : PoseidonConfigBase<Params>(std::move(base)),
        internal_diagonal_minus_one(std::move(internal_diagonal_minus_one)) {}

  constexpr static Poseidon2Config CreateDefault() {
    // TODO(chokobole): Only |BabyBear| has both shift and diagonal arrays.
    // We assume the Plonky3 team developed the concept of the shift array,
    // and will use it in most cases, so we set it as the default.
    // Other small prime fields have only a shift array, while larger prime
    // fields have only a diagonal array. For more details, refer to
    // tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_xxx.h.
    if constexpr (Params::kInternalMatrixVendor == Poseidon2Vendor::kPlonky3 &&
                  PrimeField::Config::kModulusBits <= 32) {
      return Create(GetPoseidon2InternalShiftArray<Params>());
    } else {
      return Create(GetPoseidon2InternalDiagonalArray<Params>());
    }
  }

  constexpr static Poseidon2Config CreateDefault(math::Matrix<F>&& ark) {
    static_assert(Params::kInternalMatrixVendor == Poseidon2Vendor::kPlonky3);
    return Create(GetPoseidon2InternalShiftArray<Params>(), std::move(ark));
  }

  constexpr static Poseidon2Config Create(
      const std::array<PrimeField, Params::kWidth>&
          internal_diagonal_minus_one) {
    constexpr Poseidon2ConfigEntry config_entry(Params::kRate, Params::kAlpha,
                                                Params::kFullRounds,
                                                Params::kPartialRounds);
    Poseidon2Config ret = config_entry.ToPoseidon2Config<Params>();
    ret.internal_diagonal_minus_one = math::Vector<F>(Params::kWidth);
    for (size_t i = 0; i < Params::kWidth; ++i) {
      if constexpr (math::FiniteFieldTraits<F>::kIsPackedPrimeField) {
        ret.internal_diagonal_minus_one[i] =
            F::Broadcast(internal_diagonal_minus_one[i]);
      } else {
        ret.internal_diagonal_minus_one[i] = internal_diagonal_minus_one[i];
      }
    }
    FindPoseidon2ARK(config_entry.ToPoseidonGrainLFSRConfig<F>(), ret.ark);
    return ret;
  }

  constexpr static Poseidon2Config Create(
      const std::array<uint8_t, Params::kRate>& internal_shifts) {
    Poseidon2ConfigEntry config_entry(Params::kRate, Params::kAlpha,
                                      Params::kFullRounds,
                                      Params::kPartialRounds);
    math::Matrix<F> ark;
    FindPoseidon2ARK(config_entry.ToPoseidonGrainLFSRConfig<F>(), ark);
    return Create(config_entry, internal_shifts, std::move(ark));
  }

  // NOTE(chokobole): If another variant method that accepts ark is added,
  // remember to update the code in icicle_poseidon2_holder.h as well.
  constexpr static Poseidon2Config Create(
      const std::array<uint8_t, Params::kRate>& internal_shifts,
      math::Matrix<F>&& ark) {
    Poseidon2ConfigEntry config_entry(Params::kRate, Params::kAlpha,
                                      Params::kFullRounds,
                                      Params::kPartialRounds);
    return Create(config_entry, internal_shifts, std::move(ark));
  }

 private:
  constexpr static Poseidon2Config Create(
      const Poseidon2ConfigEntry& config_entry,
      const std::array<uint8_t, Params::kRate>& internal_shifts,
      math::Matrix<F>&& ark) {
    Poseidon2Config ret = config_entry.ToPoseidon2Config<Params>();
    if constexpr (math::FiniteFieldTraits<F>::kIsPackedPrimeField) {
      ret.internal_diagonal_minus_one = math::Vector<F>(Params::kWidth);
      ret.internal_diagonal_minus_one[0] = F(PrimeField::Config::kModulus - 2);
      for (size_t i = 1; i < Params::kWidth; ++i) {
        ret.internal_diagonal_minus_one[i] =
            F(uint32_t{1} << internal_shifts[i - 1]);
      }
    } else {
      ret.internal_shifts = math::Vector<uint8_t>(Params::kRate);
      for (size_t i = 0; i < Params::kRate; ++i) {
        ret.internal_shifts[i] = internal_shifts[i];
      }
    }
    ret.ark = std::move(ark);
    return ret;
  }
};

template <typename Params>
Poseidon2Config<Params> Poseidon2ConfigEntry::ToPoseidon2Config() const {
  return Poseidon2Config<Params>();
}

}  // namespace crypto

namespace base {

template <typename Params>
class Copyable<crypto::Poseidon2Config<Params>> {
 public:
  using F = typename Params::Field;
  static bool WriteTo(const crypto::Poseidon2Config<Params>& config,
                      Buffer* buffer) {
    return Copyable<crypto::PoseidonConfigBase<Params>>::WriteTo(config,
                                                                 buffer) &&
           buffer->WriteMany(config.internal_diagonal_minus_one);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::Poseidon2Config<Params>* config) {
    crypto::PoseidonConfigBase<Params> base;
    math::Matrix<F> internal_diagonal_minus_one;
    if (!buffer.ReadMany(&base, &internal_diagonal_minus_one)) {
      return false;
    }

    *config = {std::move(base), std::move(internal_diagonal_minus_one)};
    return true;
  }

  static size_t EstimateSize(const crypto::Poseidon2Config<Params>& config) {
    const crypto::PoseidonConfigBase<Params>& base =
        static_cast<const crypto::PoseidonConfigBase<Params>&>(config);
    return base::EstimateSize(base, config.internal_diagonal_minus_one);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_H_
