#include "tachyon/c/crypto/random/rng.h"

#include <stdio.h>

int main() {
  uint8_t seed[] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};
  size_t seed_size = sizeof(seed) / sizeof(seed[0]);

  tachyon_rng* rng =
      tachyon_rng_create_from_seed(TACHYON_RNG_XOR_SHIFT, seed, seed_size);

  if (!rng) {
    printf("Failed to create RNG.\n");
    return 1;
  }

  uint32_t rand_u32 = tachyon_rng_get_next_u32(rng);
  uint64_t rand_u64 = tachyon_rng_get_next_u64(rng);
  printf("Next uint32: %u\n", rand_u32);
  printf("Next uint64: %lu\n", rand_u64);

  tachyon_rng_destroy(rng);
  return 0;
}
