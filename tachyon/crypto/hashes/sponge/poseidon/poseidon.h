// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/crypto/hashes/prime_field_serializable.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config.h"
#include "tachyon/crypto/hashes/sponge/sponge.h"

namespace tachyon::crypto {

// Poseidon Sponge Hash: Absorb → Permute → Squeeze
// Absorb: Absorb elements into the sponge.
// Permute: Transform the |state| using a series of operations.
//   1. Apply ARK (addition of round constants) to |state|.
//   2. Apply S-Box (xᵅ) to |state|.
//   3. Apply MDS matrix to |state|.
// Squeeze: Squeeze elements out of the sponge.
// This implementation of Poseidon is entirely Fractal's implementation in
// [COS20][cos] with small syntax changes. See https://eprint.iacr.org/2019/1076
template <typename PrimeFieldTy>
struct PoseidonSponge
    : public FieldBasedCryptographicSponge<PoseidonSponge<PrimeFieldTy>> {
  using F = PrimeFieldTy;

  struct State {
    // Current sponge's state (current elements in the permutation block)
    math::Vector<F> elements;

    // Current mode (whether its absorbing or squeezing)
    DuplexSpongeMode mode = DuplexSpongeMode::Absorbing();

    State() = default;
    explicit State(size_t size) : elements(size) {
      for (size_t i = 0; i < size; ++i) {
        elements[i] = F::Zero();
      }
    }

    size_t size() const { return elements.size(); }

    F& operator[](size_t idx) { return elements[idx]; }
    const F& operator[](size_t idx) const { return elements[idx]; }
  };

  // Sponge Config
  PoseidonConfig<F> config;

  // Sponge State
  State state;

  PoseidonSponge() = default;
  explicit PoseidonSponge(const PoseidonConfig<F>& config)
      : config(config), state(config.rate + config.capacity) {}
  PoseidonSponge(const PoseidonConfig<F>& config, const State& state)
      : config(config), state(state) {}
  PoseidonSponge(const PoseidonConfig<F>& config, State&& state)
      : config(config), state(std::move(state)) {}

  void ApplySBox(bool is_full_round) {
    if (is_full_round) {
      // Full rounds apply the S-Box (xᵅ) to every element of |state|.
      for (F& elem : state.elements) {
        elem = elem.Pow(math::BigInt<1>(config.alpha));
      }
    } else {
      // Partial rounds apply the S-Box (xᵅ) to just the first element of
      // |state|.
      state[0] = state[0].Pow(math::BigInt<1>(config.alpha));
    }
  }

  void ApplyARK(Eigen::Index round_number) {
    state.elements += config.ark.row(round_number);
  }

  void ApplyMDS() { state.elements = config.mds * state.elements; }

  void Permute() {
    size_t full_rounds_over_2 = config.full_rounds / 2;
    for (size_t i = 0; i < full_rounds_over_2; ++i) {
      ApplyARK(i);
      ApplySBox(true);
      ApplyMDS();
    }
    for (size_t i = full_rounds_over_2;
         i < full_rounds_over_2 + config.partial_rounds; ++i) {
      ApplyARK(i);
      ApplySBox(false);
      ApplyMDS();
    }
    for (size_t i = full_rounds_over_2 + config.partial_rounds;
         i < config.partial_rounds + config.full_rounds; ++i) {
      ApplyARK(i);
      ApplySBox(true);
      ApplyMDS();
    }
  }

  // Absorbs everything in |elements|, this does not end in an absorbing.
  void AbsorbInternal(size_t rate_start_index, const std::vector<F>& elements) {
    size_t elements_idx = 0;
    while (true) {
      size_t remaining_size = elements.size() - elements_idx;
      // if we can finish in this call
      if (rate_start_index + remaining_size <= config.rate) {
        for (size_t i = 0; i < remaining_size; ++i, ++elements_idx) {
          state[config.capacity + i + rate_start_index] +=
              elements[elements_idx];
        }
        state.mode.type = DuplexSpongeMode::Type::kAbsorbing;
        state.mode.next_index = rate_start_index + remaining_size;
        break;
      }
      // otherwise absorb (|config.rate| - |rate_start_index|) elements
      size_t num_elements_absorbed = config.rate - rate_start_index;
      for (size_t i = 0; i < num_elements_absorbed; ++i, ++elements_idx) {
        state[config.capacity + i + rate_start_index] += elements[elements_idx];
      }
      Permute();
      rate_start_index = 0;
    }
  }

  // Squeeze |output| many elements. This does not end in a squeezing.
  void SqueezeInternal(size_t rate_start_index, std::vector<F>* output) {
    size_t output_size = output->size();
    size_t output_idx = 0;
    while (true) {
      size_t output_remaining_size = output_size - output_idx;
      // if we can finish in this call
      if (rate_start_index + output_remaining_size <= config.rate) {
        for (size_t i = 0; i < output_remaining_size; ++i) {
          (*output)[output_idx + i] =
              state[config.capacity + rate_start_index + i];
        }
        state.mode.type = DuplexSpongeMode::Type::kSqueezing;
        state.mode.next_index = rate_start_index + output_remaining_size;
        return;
      }

      // otherwise squeeze (|config.rate| - |rate_start_index|) elements
      size_t num_elements_squeezed = config.rate - rate_start_index;
      for (size_t i = 0; i < num_elements_squeezed; ++i) {
        (*output)[output_idx + i] =
            state[config.capacity + rate_start_index + i];
      }

      if (output_remaining_size != config.rate) {
        Permute();
      }
      output_idx += num_elements_squeezed;
      rate_start_index = 0;
    }
  }

  // CryptographicSponge methods
  template <typename T>
  bool Absorb(const T& input) {
    std::vector<F> elements;
    if (!SerializeToFieldElements(input, &elements)) return false;

    switch (state.mode.type) {
      case DuplexSpongeMode::Type::kAbsorbing: {
        size_t absorb_index = state.mode.next_index;
        if (absorb_index == config.rate) {
          Permute();
          absorb_index = 0;
        }
        AbsorbInternal(absorb_index, elements);
        return true;
      }
      case DuplexSpongeMode::Type::kSqueezing: {
        Permute();
        AbsorbInternal(0, elements);
        return true;
      }
    }
    NOTREACHED();
    return false;
  }

  std::vector<uint8_t> SqueezeBytes(size_t num_bytes) {
    size_t usable_bytes = (F::kModulusBits - 1) / 8;

    size_t num_elements = (num_bytes + usable_bytes - 1) / usable_bytes;
    std::vector<F> src_elements = SqueezeNativeFieldElements(num_elements);

    std::vector<F> bytes;
    bytes.reserve(usable_bytes * num_elements);
    for (const F& elem : src_elements) {
      std::vector<uint8_t> elem_bytes = elem.ToBigInt().ToBytesLE();
      bytes.insert(bytes.end(), elem_bytes.begin(), elem_bytes.end());
    }

    bytes.resize(num_bytes);
    return bytes;
  }

  std::vector<bool> SqueezeBits(size_t num_bits) {
    size_t usable_bits = F::kModulusBits - 1;

    size_t num_elements = (num_bits + usable_bits - 1) / usable_bits;
    std::vector<F> src_elements = SqueezeNativeFieldElements(num_elements);

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
      const std::vector<FieldElementSize>& sizes) {
    if constexpr (F::Characteristic() == F2::Characteristic()) {
      // native case
      return this->SqueezeNativeFieldElementsWithSizes(sizes);
    }
    return this->template SqueezeFieldElementsWithSizesDefaultImpl<F2>(sizes);
  }

  template <typename F2 = F>
  std::vector<F2> SqueezeFieldElements(size_t num_elements) {
    if constexpr (std::is_same_v<F, F2>) {
      return SqueezeNativeFieldElements(num_elements);
    } else {
      return SqueezeFieldElementsWithSizes<F2>(base::CreateVector(
          num_elements, []() { return FieldElementSize::Full(); }));
    }
  }

  // FieldBasedCryptographicSponge methods
  std::vector<F> SqueezeNativeFieldElements(size_t num_elements) {
    std::vector<F> ret =
        base::CreateVector(num_elements, []() { return F::Zero(); });
    switch (state.mode.type) {
      case DuplexSpongeMode::Type::kAbsorbing: {
        Permute();
        SqueezeInternal(0, &ret);
        return ret;
      }
      case DuplexSpongeMode::Type::kSqueezing: {
        size_t squeeze_index = state.mode.next_index;
        if (squeeze_index == config.rate) {
          Permute();
          squeeze_index = 0;
        }
        SqueezeInternal(squeeze_index, &ret);
        return ret;
      }
    }
    NOTREACHED();
    return {};
  }
};

template <typename PrimeFieldTy>
struct CryptographicSpongeTraits<PoseidonSponge<PrimeFieldTy>> {
  using F = PrimeFieldTy;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
