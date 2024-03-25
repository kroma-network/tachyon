#ifndef VENDORS_CIRCOM_BENCHMARK_BIT_CONVERSION_H_
#define VENDORS_CIRCOM_BENCHMARK_BIT_CONVERSION_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "absl/types/span.h"

namespace tachyon::circom {

template <typename F>
std::vector<F> Uint8ToBitVector(absl::Span<const uint8_t> uint8_vec) {
  std::vector<F> bit_vec;
  bit_vec.reserve(uint8_vec.size() * 8);
  for (size_t i = 0; i < uint8_vec.size(); i++) {
    for (size_t j = 0; j < 8; ++j) {
      bit_vec.push_back(F((uint8_vec[i] >> (7 - j)) & 1));
    }
  }
  return bit_vec;
}

template <typename F>
std::vector<uint8_t> BitToUint8Vector(absl::Span<const F> bit_vec) {
  std::vector<uint8_t> uint8_vec;
  size_t size = (bit_vec.size() + 7) / 8;
  uint8_vec.resize(size, 0);
  for (size_t i = 0; i < bit_vec.size(); i++) {
    size_t idx = i / 8;
    uint8_vec[idx] |= (bit_vec[i].ToBigInt()[0] << (7 - (i % 8)));
  }
  return uint8_vec;
}

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_BENCHMARK_BIT_CONVERSION_H_
