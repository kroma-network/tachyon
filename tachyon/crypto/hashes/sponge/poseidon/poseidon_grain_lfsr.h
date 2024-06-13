// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_GRAIN_LFSR_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_GRAIN_LFSR_H_

#include <bitset>

#include "tachyon/export.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::crypto {

struct TACHYON_EXPORT PoseidonGrainLFSRConfig {
  bool is_sbox_an_inverse = false;
  uint64_t prime_num_bits = 0;
  uint64_t state_len = 0;
  uint64_t num_full_rounds = 0;
  uint64_t num_partial_rounds = 0;
};

// GrainLFSR is a pseudo-random generator using a stream cipher.
// It is used to generate ARK and MDS for Poseidon.
template <typename _PrimeField>
struct PoseidonGrainLFSR {
  using PrimeField = _PrimeField;

  uint64_t prime_num_bits = 0;
  bool state[80] = {
      false,
  };
  size_t head = 0;

  explicit PoseidonGrainLFSR(const PoseidonGrainLFSRConfig& config);

  std::bitset<PrimeField::kModulusBits> GetBits(size_t num_bits);

  math::Vector<PrimeField> GetFieldElementsRejectionSampling(size_t num_elems);

  math::Vector<PrimeField> GetFieldElementsModP(size_t num_elems);

 private:
  bool Update() {
    bool new_bit = state[(head + 62) % 80] ^ state[(head + 51) % 80] ^
                   state[(head + 38) % 80] ^ state[(head + 23) % 80] ^
                   state[(head + 13) % 80] ^ state[head];

    state[head] = new_bit;
    head = (head + 1) % 80;

    return new_bit;
  }

  void Init() {
    for (size_t i = 0; i < 160; ++i) {
      Update();
    }
  }
};

template <typename PrimeField>
PoseidonGrainLFSR<PrimeField>::PoseidonGrainLFSR(
    const PoseidonGrainLFSRConfig& config)
    : prime_num_bits(config.prime_num_bits) {
  std::fill(std::begin(state), std::end(state), false);

  // b0, b1 describe the field
  state[1] = true;

  // b2, ..., b5 describe the S-BOX
  state[5] = config.is_sbox_an_inverse;

  auto fill_state = [&](uint64_t value, size_t start, size_t end) {
    for (size_t i = end; i >= start && i <= end; --i) {
      state[i] = value & 1;
      value >>= 1;
    }
  };

  // b6, ..., b17 are the binary representation of n (prime_num_bits)
  fill_state(prime_num_bits, 6, 17);

  // b18, ..., b29 are the binary representation of t
  // state_len, rate + capacity
  fill_state(config.state_len, 18, 29);

  // b30, ..., b39 are the binary representation of R_F (num_full_rounds)
  fill_state(config.num_full_rounds, 30, 39);

  // b40, ..., b49 are the binary representation of R_P (num_partial_rounds)
  fill_state(config.num_partial_rounds, 40, 49);

  // b50, ..., b79 are set to 1
  for (size_t i = 50; i < 80; ++i) {
    state[i] = true;
  }

  head = 0;
  Init();
}

template <typename PrimeField>
std::bitset<PrimeField::kModulusBits> PoseidonGrainLFSR<PrimeField>::GetBits(
    size_t num_bits) {
  std::bitset<PrimeField::kModulusBits> ret;
  for (size_t i = 0; i < num_bits; ++i) {
    // Obtain the first bit
    bool new_bit = Update();

    // Loop until the first bit is 1
    while (!new_bit) {
      // Discard the second bit
      Update();
      // Obtain another first bit
      new_bit = Update();
    }

    // Obtain the second bit
    ret.set(i, Update());
  }
  return ret;
}

// Rejects elements greater than the modulus and resamples.
template <typename PrimeField>
math::Vector<PrimeField>
PoseidonGrainLFSR<PrimeField>::GetFieldElementsRejectionSampling(
    size_t num_elems) {
  using BigInt = typename PrimeField::BigIntTy;

  CHECK_EQ(PrimeField::Config::kModulusBits, prime_num_bits);

  math::Vector<PrimeField> ret(num_elems);

  for (size_t i = 0; i < num_elems; ++i) {
    // Perform rejection sampling
    while (true) {
      // Obtain n bits and make it most-significant-bit first
      std::bitset<PrimeField::kModulusBits> bits = GetBits(prime_num_bits);
      BigInt bigint = BigInt::FromBitsBE(bits);

      if (bigint < BigInt(PrimeField::Config::kModulus)) {
        ret[i] = PrimeField::FromBigInt(bigint);
        break;
      }
    }
  }

  return ret;
}

// Samples n bits and computes the remainder modulo P.
template <typename PrimeField>
math::Vector<PrimeField> PoseidonGrainLFSR<PrimeField>::GetFieldElementsModP(
    size_t num_elems) {
  using BigInt = typename PrimeField::BigIntTy;

  CHECK_EQ(PrimeField::Config::kModulusBits, prime_num_bits);

  math::Vector<PrimeField> ret(num_elems);

  for (size_t i = 0; i < num_elems; ++i) {
    // Obtain n bits and make it most-significant-bit first
    std::bitset<PrimeField::kModulusBits> bits = GetBits(prime_num_bits);

    BigInt bigint = BigInt::FromBitsBE(bits);
    bigint %= BigInt(PrimeField::Config::kModulus);

    ret[i] = PrimeField::FromBigInt(bigint);
  }

  return ret;
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_GRAIN_LFSR_H_
