#ifndef TACHYON_CRYPTO_RANDOM_CHA_CHA20_CHA_CHA20_RNG_H_
#define TACHYON_CRYPTO_RANDOM_CHA_CHA20_CHA_CHA20_RNG_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "absl/base/internal/endian.h"
#include "openssl/chacha.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/random.h"
#include "tachyon/crypto/random/block_rng.h"
#include "tachyon/export.h"

namespace tachyon::crypto {

// ChaCha20RNG stands for "ChaCha20 Random Number Generator".
// Please use |base::Uniform()| mostly for production. This is being used in
// snark-verifier and we need a seed stability to ensure the same random number
// to be generated from a given seed.
//
// See
// [Seed Stability]
// (https://abseil.io/docs/cpp/guides/random#classes-of-generator-stability),
// [ChaCha20 and Poly1305 for IETF Protocols]
// (https://tools.ietf.org/html/rfc8439) and
// [rand_chacha](https://github.com/rust-random/rand/tree/master/rand_chacha).
class TACHYON_EXPORT ChaCha20RNG final : public BlockRNG<ChaCha20RNG> {
 public:
  constexpr static size_t kSeedSize = 32;
  constexpr static size_t kInputSize = 64;
  constexpr static size_t kOutputSize = 64;
  constexpr static size_t kStateSize =
      sizeof(size_t) + kInputSize + kOutputSize;
  constexpr static uint8_t kDefaultSeed[kSeedSize] = {
      0,
  };

  ChaCha20RNG() : ChaCha20RNG(kDefaultSeed) {}

  // RNG methods
  void SetRandomSeed() override {
    uint8_t seed[kSeedSize];
    for (size_t i = 0; i < 4; ++i) {
      *reinterpret_cast<uint64_t*>(&seed[i * 8]) =
          base::Uniform(base::Range<uint64_t>::All());
    }
    CHECK(SetSeed(seed));
  }

  [[nodiscard]] bool SetSeed(absl::Span<const uint8_t> seed) override {
    if (seed.size() != kSeedSize) {
      LOG(ERROR) << "Seed size must be " << kSeedSize;
      return false;
    }
    for (size_t i = 0; i < 8; ++i) {
      input_[i + 4] = absl::little_endian::Load32(&seed[4 * i]);
    }
    return true;
  }

  [[nodiscard]] bool ReadFromBuffer(
      const base::ReadOnlyBuffer& buffer) override {
    size_t index;
    uint8_t input[kInputSize];
    uint8_t output[kOutputSize];
    base::EndianAutoReset auto_reset(buffer, base::Endian::kLittle);
    if (!buffer.ReadMany(&index, input, output)) return false;
    index_ = index;
    memcpy(input_, input, kInputSize);
    memcpy(output_, output, kOutputSize);
    return true;
  }

  [[nodiscard]] bool WriteToBuffer(base::Buffer& buffer) const override {
    base::EndianAutoReset auto_reset(buffer, base::Endian::kLittle);
    return buffer.WriteMany(index_, input_, output_);
  }

 private:
  friend class BlockRNG<ChaCha20RNG>;

  explicit ChaCha20RNG(const uint8_t seed[kSeedSize]) {
    // constants
    input_[0] = 0x61707865;
    input_[1] = 0x3320646E;
    input_[2] = 0x79622D32;
    input_[3] = 0x6B206574;

    // seed
    CHECK(SetSeed(absl::Span<const uint8_t>(seed, kSeedSize)));

    // counter
    input_[12] = 0;
    input_[13] = 0;
    input_[14] = 0;
    input_[15] = 0;
  }

  ChaCha20RNG(size_t index, const uint8_t input[kInputSize],
              const uint8_t output[kOutputSize])
      : BlockRNG<ChaCha20RNG>(index) {
    memcpy(input_, input, kInputSize);
    memcpy(output_, output, kOutputSize);
  }

  // BlockRNG<ChaCha20RNG> methods
  void Update() {
    chacha_core(output_, input_);
    if (++input_[12] == 0) {
      ++input_[13];
    }
  }

  uint32_t Get(size_t index) const {
    return reinterpret_cast<const uint32_t*>(output_)[index];
  }

  uint32_t input_[kInputSize / sizeof(uint32_t)];
  uint8_t output_[kOutputSize];
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_RANDOM_CHA_CHA20_CHA_CHA20_RNG_H_
