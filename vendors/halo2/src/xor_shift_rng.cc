#include "vendors/halo2/include/xor_shift_rng.h"

#include <string.h>

#include "tachyon/base/logging.h"
#include "tachyon/c/crypto/random/rng.h"
#include "tachyon/rs/base/container_util.h"

namespace tachyon::halo2_api {

constexpr size_t kSeedSize = 16;
constexpr size_t kStateSize = 16;

class XORShiftRng::Impl {
 public:
  explicit Impl(std::array<uint8_t, kSeedSize> seed) {
    uint8_t seed_copy[kSeedSize];
    memcpy(seed_copy, seed.data(), kSeedSize);
    rng_ = tachyon_rng_create_from_seed(TACHYON_RNG_XOR_SHIFT, seed_copy,
                                        kSeedSize);
  }
  Impl(const Impl& other) {
    uint8_t state[kStateSize];
    size_t state_len;
    tachyon_rng_get_state(other.rng_, state, &state_len);
    CHECK_EQ(state_len, kStateSize);
    rng_ =
        tachyon_rng_create_from_state(TACHYON_RNG_XOR_SHIFT, state, kStateSize);
  }
  ~Impl() { tachyon_rng_destroy(rng_); }

  uint32_t NextUint32() { return tachyon_rng_get_next_u32(rng_); }

  rust::Vec<uint8_t> GetState() const {
    uint8_t state[kStateSize];
    size_t state_len;
    tachyon_rng_get_state(rng_, state, &state_len);
    CHECK_EQ(state_len, kStateSize);
    return rs::ConvertCppContainerToRustVec(state);
  }

 private:
  tachyon_rng* rng_;
};

XORShiftRng::XORShiftRng(std::array<uint8_t, kSeedSize> seed)
    : impl_(new Impl(seed)) {}

uint32_t XORShiftRng::next_u32() { return impl_->NextUint32(); }

std::unique_ptr<XORShiftRng> XORShiftRng::clone() const {
  std::unique_ptr<XORShiftRng> ret(new XORShiftRng());
  ret->impl_.reset(new Impl(*impl_));
  return ret;
}

rust::Vec<uint8_t> XORShiftRng::state() const { return impl_->GetState(); }

std::unique_ptr<XORShiftRng> new_xor_shift_rng(
    std::array<uint8_t, kSeedSize> seed) {
  return std::make_unique<XORShiftRng>(seed);
}

}  // namespace tachyon::halo2_api
