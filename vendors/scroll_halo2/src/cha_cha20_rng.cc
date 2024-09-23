#include "vendors/scroll_halo2/include/cha_cha20_rng.h"

#include <string.h>

#include "tachyon/base/logging.h"
#include "tachyon/rs/base/container_util.h"

namespace tachyon::halo2_api {

ChaCha20Rng::ChaCha20Rng(std::array<uint8_t, kSeedSize> seed) {
  uint8_t seed_copy[kSeedSize];
  memcpy(seed_copy, seed.data(), kSeedSize);
  rng_ =
      tachyon_rng_create_from_seed(TACHYON_RNG_CHA_CHA20, seed_copy, kSeedSize);
}

ChaCha20Rng::~ChaCha20Rng() { tachyon_rng_destroy(rng_); }

uint32_t ChaCha20Rng::next_u32() { return tachyon_rng_get_next_u32(rng_); }

std::unique_ptr<ChaCha20Rng> ChaCha20Rng::clone() const {
  uint8_t state[kStateSize];
  size_t state_len;
  tachyon_rng_get_state(rng_, state, &state_len);
  CHECK_EQ(state_len, kStateSize);
  tachyon_rng* rng =
      tachyon_rng_create_from_state(TACHYON_RNG_CHA_CHA20, state, kStateSize);
  return std::make_unique<ChaCha20Rng>(rng);
}

rust::Vec<uint8_t> ChaCha20Rng::state() const {
  rust::Vec<uint8_t> ret = rs::CreateEmptyVector<uint8_t>(kStateSize);
  size_t state_len;
  tachyon_rng_get_state(rng_, ret.data(), &state_len);
  CHECK_EQ(state_len, kStateSize);
  return ret;
}

std::unique_ptr<ChaCha20Rng> new_cha_cha20_rng(
    std::array<uint8_t, ChaCha20Rng::kSeedSize> seed) {
  return std::make_unique<ChaCha20Rng>(seed);
}

}  // namespace tachyon::halo2_api
