#ifndef TACHYON_CRYPTO_RANDOM_XOR_SHIFT_XOR_SHIFT_RNG_H_
#define TACHYON_CRYPTO_RANDOM_XOR_SHIFT_XOR_SHIFT_RNG_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/random.h"
#include "tachyon/crypto/random/rng.h"
#include "tachyon/export.h"

namespace tachyon::crypto {

// XORShiftRNG stands for "XOR Shift Random Number Generator".
// Please use |base::Uniform()| mostly for production. This is being used in
// Halo2 and we need a seed stability to ensure the same random number to be
// generated from a given seed.
//
// See
// [Seed Stability]
// (https://abseil.io/docs/cpp/guides/random#classes-of-generator-stability)
// and [Xorshift RNGs](https://www.jstatsoft.org/v08/i14/paper).
class TACHYON_EXPORT XORShiftRNG final : public RNG {
 public:
  constexpr static size_t kSeedSize = 16;
  constexpr static size_t kStateSize = 16;

  XORShiftRNG() = default;

  // RNG methods
  void SetRandomSeed() override {
    uint8_t seed[kSeedSize];
    *reinterpret_cast<uint64_t*>(&seed[0]) =
        base::Uniform(base::Range<uint64_t>::All());
    *reinterpret_cast<uint64_t*>(&seed[8]) =
        base::Uniform(base::Range<uint64_t>::All());
    CHECK(SetSeed(seed));
  }

  [[nodiscard]] bool SetSeed(absl::Span<const uint8_t> seed) override {
    if (seed.size() != kSeedSize) {
      LOG(ERROR) << "Seed size must be " << kSeedSize;
      return false;
    }
    memcpy(&x_, &seed[0], sizeof(uint32_t));
    memcpy(&y_, &seed[4], sizeof(uint32_t));
    memcpy(&z_, &seed[8], sizeof(uint32_t));
    memcpy(&w_, &seed[12], sizeof(uint32_t));
    return true;
  }

  uint32_t NextUint32() override {
    uint32_t t = x_ ^ (x_ << 11);
    x_ = y_;
    y_ = z_;
    z_ = w_;
    w_ = w_ ^ (w_ >> 19) ^ (t ^ (t >> 8));
    return w_;
  }

  [[nodiscard]] bool ReadFromBuffer(
      const base::ReadOnlyBuffer& buffer) override {
    uint32_t x, y, z, w;
    base::EndianAutoReset auto_reset(buffer, base::Endian::kLittle);
    if (!buffer.ReadMany(&x, &y, &z, &w)) return false;
    x_ = x;
    y_ = y;
    z_ = z;
    w_ = w;
    return true;
  }

  [[nodiscard]] bool WriteToBuffer(base::Buffer& buffer) const override {
    base::EndianAutoReset auto_reset(buffer, base::Endian::kLittle);
    return buffer.WriteMany(x_, y_, z_, w_);
  }

 private:
  XORShiftRNG(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
      : x_(x), y_(y), z_(z), w_(w) {}

  uint32_t x_ = 195911405;  // 0xBAD_5EED
  uint32_t y_ = 195911405;  // 0xBAD_5EED
  uint32_t z_ = 195911405;  // 0xBAD_5EED
  uint32_t w_ = 195911405;  // 0xBAD_5EED
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_RANDOM_XOR_SHIFT_XOR_SHIFT_RNG_H_
