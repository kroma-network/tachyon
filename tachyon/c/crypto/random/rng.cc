#include "tachyon/c/crypto/random/rng.h"

#include "absl/types/span.h"

#include "tachyon/base/logging.h"
#include "tachyon/crypto/random/cha_cha20/cha_cha20_rng.h"
#include "tachyon/crypto/random/rng_type.h"
#include "tachyon/crypto/random/xor_shift/xor_shift_rng.h"

using namespace tachyon;

namespace {

void CheckKnownType(uint8_t type) {
  CHECK(type == TACHYON_RNG_XOR_SHIFT || type == TACHYON_RNG_CHA_CHA20);
}

tachyon_rng* CreateFromType(uint8_t type) {
  tachyon_rng* rng = new tachyon_rng;
  rng->type = type;
  switch (static_cast<crypto::RNGType>(type)) {
    case crypto::RNGType::kXORShift:
      rng->extra = new crypto::XORShiftRNG;
      return rng;
    case crypto::RNGType::kChaCha20:
      rng->extra = new crypto::ChaCha20RNG;
      return rng;
  }
  NOTREACHED();
  return rng;
}

size_t GetStateLen(uint8_t type) {
  switch (static_cast<crypto::RNGType>(type)) {
    case crypto::RNGType::kXORShift:
      return crypto::XORShiftRNG::kStateSize;
    case crypto::RNGType::kChaCha20:
      return crypto::ChaCha20RNG::kStateSize;
  }
  NOTREACHED();
  return 0;
}

}  // namespace

tachyon_rng* tachyon_rng_create_from_seed(uint8_t type, const uint8_t* seed,
                                          size_t seed_len) {
  tachyon_rng* rng = CreateFromType(type);
  CHECK(reinterpret_cast<crypto::RNG*>(rng->extra)
            ->SetSeed(absl::Span<const uint8_t>(seed, seed_len)));
  return rng;
}

tachyon_rng* tachyon_rng_create_from_state(uint8_t type, const uint8_t* state,
                                           size_t state_len) {
  tachyon_rng* rng = CreateFromType(type);
  base::ReadOnlyBuffer buffer(state, state_len);
  CHECK(reinterpret_cast<crypto::RNG*>(rng->extra)->ReadFromBuffer(buffer));
  return rng;
}

void tachyon_rng_destroy(tachyon_rng* rng) {
  CheckKnownType(rng->type);
  delete reinterpret_cast<crypto::RNG*>(rng->extra);
  delete rng;
}

uint32_t tachyon_rng_get_next_u32(tachyon_rng* rng) {
  CheckKnownType(rng->type);
  return reinterpret_cast<crypto::RNG*>(rng->extra)->NextUint32();
}

uint64_t tachyon_rng_get_next_u64(tachyon_rng* rng) {
  CheckKnownType(rng->type);
  return reinterpret_cast<crypto::RNG*>(rng->extra)->NextUint64();
}

void tachyon_rng_get_state(const tachyon_rng* rng, uint8_t* state,
                           size_t* state_len) {
  *state_len = GetStateLen(rng->type);
  if (state == nullptr) return;
  base::Buffer buffer(state, *state_len);
  CHECK(reinterpret_cast<crypto::RNG*>(rng->extra)->WriteToBuffer(buffer));
}
