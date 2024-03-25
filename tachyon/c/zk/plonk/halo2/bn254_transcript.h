/**
 * @file
 * @brief Defines the interface for the transcript writer used within the Halo2
 * bn254 proof system.
 */

#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_TRANSCRIPT_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_TRANSCRIPT_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/zk/plonk/halo2/constants.h"

struct tachyon_halo2_bn254_transcript_writer {
  uint8_t type;
  void* extra;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new transcript writer of the specified type.
 *
 * @param type The type of transcript writer to create.
 * @return A pointer to the newly created transcript writer.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_transcript_writer*
tachyon_halo2_bn254_transcript_writer_create(uint8_t type);

/**
 * @brief Creates a new transcript writer of the specified type, initializing
 * its state from the given buffer.
 *
 * @param type The type of transcript writer to create.
 * @param state A buffer containing the initial state.
 * @param state_len The length of the state buffer.
 * @return A pointer to the newly created transcript writer with its state
 * initialized.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_transcript_writer*
tachyon_halo2_bn254_transcript_writer_create_from_state(uint8_t type,
                                                        const uint8_t* state,
                                                        size_t state_len);

/**
 * @brief Destroys a transcript writer, freeing its resources.
 *
 * @param writer The transcript writer to destroy.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_transcript_writer_destroy(
    tachyon_halo2_bn254_transcript_writer* writer);

/**
 * @brief Updates the transcript with new data.
 *
 * @param writer The transcript writer to update.
 * @param data The data to add to the transcript.
 * @param data_len The length of the data.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_transcript_writer_update(
    tachyon_halo2_bn254_transcript_writer* writer, const uint8_t* data,
    size_t data_len);

/**
 * @brief Finalizes the transcript, producing a hash.
 * For Poseidon transcript types, use
 * tachyon_halo2_bn254_transcript_writer_squeeze instead.
 *
 * If |data| is NULL, then it populates |data_len| with length to be used.
 * If |data| is not NULL, then it populates |data| with the hash.
 * |tachyon_halo2_bn254_transcript_writer_squeeze| instead. Otherwise, it
 * terminates the program.
 *
 * @param writer The transcript writer to finalize.
 * @param data Buffer to store the resulting hash.
 * @param data_len Pointer to store the length of the hash or the required
 * buffer size.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_transcript_writer_finalize(
    tachyon_halo2_bn254_transcript_writer* writer, uint8_t* data,
    size_t* data_len);

/**
 * @brief Squeezes a field element from the transcript. Only valid for Poseidon
 * transcripts.
 *
 * If the type of the transcript is not poseidon, it terminates the program.
 *
 * @param writer The transcript writer.
 * @return A field element squeezed from the transcript.
 */

TACHYON_C_EXPORT tachyon_bn254_fr tachyon_halo2_bn254_transcript_writer_squeeze(
    tachyon_halo2_bn254_transcript_writer* writer);

/**
 * @brief Retrieves the internal state of the transcript writer.
 *
 * If |state| is NULL, then it populates |state_len| with length to be used.
 * If |state| is not NULL, then it populates |state| with its internal state.
 *
 * @param writer The transcript writer.
 * @param state Buffer to store the internal state.
 * @param state_len Pointer to store the length of the internal state or the
 * required buffer size.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_transcript_writer_get_state(
    const tachyon_halo2_bn254_transcript_writer* writer, uint8_t* state,
    size_t* state_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_TRANSCRIPT_H_
