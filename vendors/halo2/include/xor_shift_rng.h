#ifndef VENDORS_HALO2_INCLUDE_XOR_SHIFT_RNG_H_
#define VENDORS_HALO2_INCLUDE_XOR_SHIFT_RNG_H_

#include <stdint.h>

#include <array>
#include <memory>

#include "rust/cxx.h"

namespace tachyon::halo2_api {

class XORShiftRng {
 public:
  XORShiftRng() = default;
  explicit XORShiftRng(std::array<uint8_t, 16> seed);

  uint32_t next_u32();
  std::unique_ptr<XORShiftRng> clone() const;
  rust::Vec<uint8_t> state() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

std::unique_ptr<XORShiftRng> new_xor_shift_rng(std::array<uint8_t, 16> seed);

}  // namespace tachyon::halo2_api

#endif  // VENDORS_HALO2_INCLUDE_XOR_SHIFT_RNG_H_
