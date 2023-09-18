#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_GRAIN_LFSR_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_GRAIN_LFSR_H_

#include <cassert>
#include <sstream>
#include <vector>

#include "tachyon/math/base/big_int.h"
#include "tachyon/math/finite_fields/prime_field_forward.h"

namespace tachyon::crypto {

template <typename Config>
class PoseidonGrainLFSR;

template <typename Config>
class PoseidonGrainLFSR<math::PrimeField<Config>> {
 public:
  using PrimeFieldTy = math::PrimeField<Config>;

  uint64_t prime_num_bits;
  bool state[80];
  size_t head;

  PoseidonGrainLFSR(bool is_sbox_an_inverse, uint64_t prime_num_bits,
                    uint64_t state_len, uint64_t num_full_rounds,
                    uint64_t num_partial_rounds);

  std::vector<bool> GetBits(size_t num_bits);

  std::vector<PrimeFieldTy> GetFieldElementsRejectionSampling(size_t num_elems);

  std::vector<PrimeFieldTy> GetFieldElementsModP(size_t num_elems);

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

ALWAYS_INLINE char ConvertToHex(const std::vector<bool>& fourBits) {
  assert(fourBits.size() == 4);
  int value =
      fourBits[0] * 8 + fourBits[1] * 4 + fourBits[2] * 2 + fourBits[3] * 1;
  if (value < 10) return '0' + value;
  return 'A' + (value - 10);
}

template <typename Config>
PoseidonGrainLFSR<math::PrimeField<Config>>::PoseidonGrainLFSR(
    bool is_sbox_an_inverse, uint64_t prime_num_bits, uint64_t state_len,
    uint64_t num_full_rounds, uint64_t num_partial_rounds)
    : prime_num_bits(prime_num_bits) {
  std::fill(std::begin(state), std::end(state), false);

  // b0, b1 describe the field
  state[1] = true;

  // b2, ..., b5 describe the S-BOX
  state[5] = is_sbox_an_inverse;

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
  fill_state(state_len, 18, 29);

  // b30, ..., b39 are the binary representation of R_F (num_full_rounds)
  fill_state(num_full_rounds, 30, 39);

  // b40, ..., b49 are the binary representation of R_P (num_partial_rounds)
  fill_state(num_partial_rounds, 40, 49);

  // b50, ..., b79 are set to 1
  for (size_t i = 50; i < 80; ++i) {
    state[i] = true;
  }

  head = 0;
  Init();
}

template <typename Config>
std::vector<bool> PoseidonGrainLFSR<math::PrimeField<Config>>::GetBits(
    size_t num_bits) {
  std::vector<bool> ret;
  ret.reserve(num_bits);
  for (size_t i = 0; i < num_bits; ++i) {
    // Obtain the first bit
    bool new_bit = Update();

    // Loop until the first bit is 1
    while (!new_bit) {
      // Discard the second bit
      new_bit = Update();
      // Obtain another first bit
      new_bit = Update();
    }

    // Obtain the second bit
    ret.push_back(Update());
  }
  return ret;
}

template <typename Config>
std::vector<math::PrimeField<Config>>
PoseidonGrainLFSR<math::PrimeField<Config>>::GetFieldElementsRejectionSampling(
    size_t num_elems) {
  assert(PrimeFieldTy::kModulusBits == prime_num_bits);
  std::vector<PrimeFieldTy> ret;
  ret.reserve(num_elems);

  // Perform rejection sampling
  for (size_t i = 0; i < num_elems; ++i) {
    // Obtain n bits and make it most-significant-bit first
    std::vector<bool> bits = GetBits(PrimeFieldTy::kModulusBits);

    std::vector<uint8_t> bytes;

    // Convert bits to hex string
    while (bits.size() % 4 != 0) {
      bits.insert(bits.begin(), false);
    }
    std::stringstream ss;
    for (size_t i = 0; i < bits.size(); i += 4) {
      std::vector<bool> chunk(bits.begin() + i, bits.begin() + i + 4);
      ss << ConvertToHex(chunk);
    }
    std::string hex_str = ss.str();

    // TODO(insun35): Implement FromBits() for BigInt
    // auto bigint = math::BigInt::FromBits(bits);
    auto bigint = math::BigInt<PrimeFieldTy::N>::FromHexString(hex_str);
    bigint %= Config::kModulus;

    auto optField = PrimeFieldTy::FromBigInt(bigint);
    ret.push_back(optField);
  }

  return ret;
}

template <typename Config>
std::vector<math::PrimeField<Config>>
PoseidonGrainLFSR<math::PrimeField<Config>>::GetFieldElementsModP(
    size_t num_elems) {
  assert(PrimeFieldTy::kModulusBits == prime_num_bits);
  std::vector<PrimeFieldTy> ret;
  ret.reserve(num_elems);

  for (size_t i = 0; i < num_elems; ++i) {
    // Obtain n bits and make it most-significant-bit first
    std::vector<bool> bits = GetBits(PrimeFieldTy::kModulusBits);

    // Convert bits to hex string
    while (bits.size() % 4 != 0) {
      bits.insert(bits.begin(), false);
    }
    std::stringstream ss;
    for (size_t i = 0; i < bits.size(); i += 4) {
      std::vector<bool> chunk(bits.begin() + i, bits.begin() + i + 4);
      ss << ConvertToHex(chunk);
    }
    std::string hex_str = ss.str();

    auto big_int = math::BigInt<PrimeFieldTy::N>::FromHexString(hex_str);
    big_int %= Config::kModulus;
    auto field_element = PrimeFieldTy::FromBigInt(big_int);

    ret.push_back(field_element);
  }

  return ret;
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_GRAIN_LFSR_H_
