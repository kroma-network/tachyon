#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_TRANSCRIPT_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_TRANSCRIPT_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/plonk/halo2/constants.h"

struct tachyon_halo2_bn254_transcript_writer {
  uint8_t type;
  void* extra;
};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_halo2_bn254_transcript_writer*
tachyon_halo2_bn254_transcript_writer_create(uint8_t type);

TACHYON_C_EXPORT tachyon_halo2_bn254_transcript_writer*
tachyon_halo2_bn254_transcript_writer_create_from_state(uint8_t type,
                                                        const uint8_t* state,
                                                        size_t state_len);

TACHYON_C_EXPORT void tachyon_halo2_bn254_transcript_writer_destroy(
    tachyon_halo2_bn254_transcript_writer* writer);

TACHYON_C_EXPORT void tachyon_halo2_bn254_transcript_writer_update(
    tachyon_halo2_bn254_transcript_writer* writer, const uint8_t* data,
    size_t data_len);

// If |data| is NULL, then it populates |data_len| with length to be used.
// If |data| is not NULL, then it populates |data| with the hash.
TACHYON_C_EXPORT void tachyon_halo2_bn254_transcript_writer_finalize(
    tachyon_halo2_bn254_transcript_writer* writer, uint8_t* data,
    size_t* data_len);

// If |state| is NULL, then it populates |state_len| with length to be used.
// If |state| is not NULL, then it populates |state| with its internal state.
TACHYON_C_EXPORT void tachyon_halo2_bn254_transcript_writer_get_state(
    const tachyon_halo2_bn254_transcript_writer* writer, uint8_t* state,
    size_t* state_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_TRANSCRIPT_H_
