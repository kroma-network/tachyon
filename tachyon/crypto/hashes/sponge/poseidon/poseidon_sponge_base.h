// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_SPONGE_BASE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_SPONGE_BASE_H_

#include <vector>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/base/logging.h"
#include "tachyon/crypto/hashes/prime_field_serializable.h"
#include "tachyon/crypto/hashes/sponge/sponge.h"

namespace tachyon::crypto {

template <typename Derived>
struct PoseidonSpongeBase : public FieldBasedCryptographicSponge<Derived> {
  using Params = typename CryptographicSpongeTraits<Derived>::Params;
  using F = typename CryptographicSpongeTraits<Derived>::F;

  // CryptographicSponge methods
  template <typename T>
  bool Absorb(SpongeState<Params>& state, const T& input) const {
    if constexpr (std::is_constructible_v<absl::Span<const F>, T>) {
      return Absorb(state, absl::Span<const F>(input));
    }
    std::vector<F> elements;
    if (!SerializeToFieldElements(input, &elements)) return false;
    return Absorb(state, absl::MakeConstSpan(elements));
  }

  bool Absorb(SpongeState<Params>& state, absl::Span<const F> input) const {
    const Derived& derived = static_cast<const Derived&>(*this);

    switch (state.mode.type) {
      case DuplexSpongeMode::Type::kAbsorbing: {
        size_t absorb_index = state.mode.next_index;
        if (absorb_index == Params::kRate) {
          derived.Permute(state);
          absorb_index = 0;
        }
        AbsorbInternal(state, absorb_index, input);
        return true;
      }
      case DuplexSpongeMode::Type::kSqueezing: {
        derived.Permute(state);
        AbsorbInternal(state, 0, input);
        return true;
      }
    }
    NOTREACHED();
    return false;
  }

  std::vector<uint8_t> SqueezeBytes(SpongeState<Params>& state,
                                    size_t num_bytes) const {
    size_t usable_bytes = (F::kModulusBits - 1) / 8;

    size_t num_elements = (num_bytes + usable_bytes - 1) / usable_bytes;
    std::vector<F> src_elements =
        SqueezeNativeFieldElements(state, num_elements);

    std::vector<F> bytes;
    bytes.reserve(usable_bytes * num_elements);
    for (const F& elem : src_elements) {
      auto elem_bytes = elem.ToBigInt().ToBytesLE();
      bytes.insert(bytes.end(), elem_bytes.begin(), elem_bytes.end());
    }

    bytes.resize(num_bytes, F::Zero());
    return bytes;
  }

  std::vector<bool> SqueezeBits(SpongeState<Params>& state,
                                size_t num_bits) const {
    size_t usable_bits = F::kModulusBits - 1;

    size_t num_elements = (num_bits + usable_bits - 1) / usable_bits;
    std::vector<F> src_elements =
        SqueezeNativeFieldElements(state, num_elements);

    std::vector<bool> bits;
    for (const F& elem : src_elements) {
      std::bitset<F::kModulusBits> elem_bits =
          elem.ToBigInt().template ToBitsLE<F::kModulusBits>();
      bits.insert(bits.end(), elem_bits.begin(), elem_bits.end());
    }
    bits.resize(num_bits);
    return bits;
  }

  template <typename F2 = F>
  std::vector<F2> SqueezeFieldElementsWithSizes(
      SpongeState<Params>& state,
      const std::vector<FieldElementSize>& sizes) const {
    if constexpr (F::Characteristic() == F2::Characteristic()) {
      // native case
      return this->SqueezeNativeFieldElementsWithSizes(state, sizes);
    }
    return this->template SqueezeFieldElementsWithSizesDefaultImpl<F2>(state,
                                                                       sizes);
  }

  template <typename F2 = F>
  std::vector<F2> SqueezeFieldElements(SpongeState<Params>& state,
                                       size_t num_elements) const {
    if constexpr (std::is_same_v<F, F2>) {
      return SqueezeNativeFieldElements(state, num_elements);
    } else {
      return SqueezeFieldElementsWithSizes<F2>(
          state, std::vector<FieldElementSize>(num_elements,
                                               FieldElementSize::Full()));
    }
  }

  // FieldBasedCryptographicSponge methods
  std::vector<F> SqueezeNativeFieldElements(SpongeState<Params>& state,
                                            size_t num_elements) const {
    const Derived& derived = static_cast<const Derived&>(*this);

    // NOTE(batzor): |SqueezeInternal| will fill all the garbage values, so it
    // is safe to have it uninitialized.
    std::vector<F> ret(num_elements);
    switch (state.mode.type) {
      case DuplexSpongeMode::Type::kAbsorbing: {
        derived.Permute(state);
        SqueezeInternal(state, 0, &ret);
        return ret;
      }
      case DuplexSpongeMode::Type::kSqueezing: {
        size_t squeeze_index = state.mode.next_index;
        if (squeeze_index == Params::kRate) {
          derived.Permute(state);
          squeeze_index = 0;
        }
        SqueezeInternal(state, squeeze_index, &ret);
        return ret;
      }
    }
    NOTREACHED();
    return {};
  }

 protected:
  void ApplyARKFull(SpongeState<Params>& state,
                    Eigen::Index round_number) const {
    const Derived& derived = static_cast<const Derived&>(*this);
    const auto& row = derived.config.ark.row(round_number);
    for (size_t i = 0; i < Params::kWidth; ++i) {
      state.elements[i] += row[i];
    }
  }

  void ApplyARKPartial(SpongeState<Params>& state,
                       Eigen::Index round_number) const {
    const Derived& derived = static_cast<const Derived&>(*this);
    state.elements[0] += derived.config.ark.row(round_number)[0];
  }

  void ApplySBoxFull(SpongeState<Params>& state) const {
    // Full rounds apply the S-Box (xᵅ) to every element of |state|.
    for (F& elem : state.elements) {
      elem = elem.template ConstPow<Params::kAlpha>();
    }
  }

  void ApplySBoxPartial(SpongeState<Params>& state) const {
    // Partial rounds apply the S-Box (xᵅ) to just the first element of
    // |state|.
    state[0] = state[0].template ConstPow<Params::kAlpha>();
  }

 private:
  // Absorbs everything in |elements|, this does not end in an absorbing.
  void AbsorbInternal(SpongeState<Params>& state, size_t rate_start_index,
                      absl::Span<const F> elements) const {
    const Derived& derived = static_cast<const Derived&>(*this);
    size_t elements_idx = 0;
    while (true) {
      size_t remaining_size = elements.size() - elements_idx;
      // if we can finish in this call
      if (rate_start_index + remaining_size <= Params::kRate) {
        for (size_t i = 0; i < remaining_size; ++i, ++elements_idx) {
          state[Params::kCapacity + i + rate_start_index] +=
              elements[elements_idx];
        }
        state.mode.type = DuplexSpongeMode::Type::kAbsorbing;
        state.mode.next_index = rate_start_index + remaining_size;
        break;
      }
      // otherwise absorb (|Params::kRate| - |rate_start_index|) elements
      size_t num_elements_absorbed = Params::kRate - rate_start_index;
      for (size_t i = 0; i < num_elements_absorbed; ++i, ++elements_idx) {
        state[Params::kCapacity + i + rate_start_index] +=
            elements[elements_idx];
      }
      derived.Permute(state);
      rate_start_index = 0;
    }
  }

  // Squeeze |output| many elements. This does not end in a squeezing.
  void SqueezeInternal(SpongeState<Params>& state, size_t rate_start_index,
                       std::vector<F>* output) const {
    const Derived& derived = static_cast<const Derived&>(*this);
    size_t output_size = output->size();
    size_t output_idx = 0;
    while (true) {
      size_t output_remaining_size = output_size - output_idx;
      // if we can finish in this call
      if (rate_start_index + output_remaining_size <= Params::kRate) {
        for (size_t i = 0; i < output_remaining_size; ++i) {
          (*output)[output_idx + i] =
              state[Params::kCapacity + rate_start_index + i];
        }
        state.mode.type = DuplexSpongeMode::Type::kSqueezing;
        state.mode.next_index = rate_start_index + output_remaining_size;
        return;
      }

      // otherwise squeeze (|Params::kRate| - |rate_start_index|) elements
      size_t num_elements_squeezed = Params::kRate - rate_start_index;
      for (size_t i = 0; i < num_elements_squeezed; ++i) {
        (*output)[output_idx + i] =
            state[Params::kCapacity + rate_start_index + i];
      }

      if (output_remaining_size != Params::kRate) {
        derived.Permute(state);
      }
      output_idx += num_elements_squeezed;
      rate_start_index = 0;
    }
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_SPONGE_BASE_H_
