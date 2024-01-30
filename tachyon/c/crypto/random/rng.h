#ifndef TACHYON_C_CRYPTO_RANDOM_RNG_H_
#define TACHYON_C_CRYPTO_RANDOM_RNG_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TACHYON_RNG_XOR_SHIFT 0

struct tachyon_rng {
  uint8_t type;
  void* extra;
};

TACHYON_C_EXPORT tachyon_rng* tachyon_rng_create_from_seed(uint8_t type,
                                                           const uint8_t* seed,
                                                           size_t seed_len);

TACHYON_C_EXPORT tachyon_rng* tachyon_rng_create_from_state(
    uint8_t type, const uint8_t* state, size_t state_len);

TACHYON_C_EXPORT void tachyon_rng_destroy(tachyon_rng* rng);

TACHYON_C_EXPORT uint32_t tachyon_rng_get_next_u32(tachyon_rng* rng);

TACHYON_C_EXPORT uint64_t tachyon_rng_get_next_u64(tachyon_rng* rng);

// If |state| is NULL, then it populates |state_len| with length to be used.
// If |state| is not NULL, then it populates |state| with its internal state.
TACHYON_C_EXPORT void tachyon_rng_get_state(const tachyon_rng* rng,
                                            uint8_t* state, size_t* state_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_CRYPTO_RANDOM_RNG_H_
