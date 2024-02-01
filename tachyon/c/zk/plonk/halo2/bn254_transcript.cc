#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"

#include <string.h>

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"

using namespace tachyon;

using Blake2bWriter =
    zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>;

tachyon_halo2_bn254_transcript_writer*
tachyon_halo2_bn254_transcript_writer_create(uint8_t type) {
  tachyon_halo2_bn254_transcript_writer* writer =
      new tachyon_halo2_bn254_transcript_writer;
  writer->type = type;
  if (type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
    base::Uint8VectorBuffer write_buf;
    Blake2bWriter* blake2b = new Blake2bWriter(std::move(write_buf));
    writer->extra = blake2b;
  }
  return writer;
}

tachyon_halo2_bn254_transcript_writer*
tachyon_halo2_bn254_transcript_writer_create_from_state(uint8_t type,
                                                        const uint8_t* state,
                                                        size_t state_len) {
  tachyon_halo2_bn254_transcript_writer* writer =
      new tachyon_halo2_bn254_transcript_writer;
  writer->type = type;
  if (type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
    base::Uint8VectorBuffer write_buf;
    Blake2bWriter* blake2b = new Blake2bWriter(std::move(write_buf));
    blake2b->SetState(absl::Span<const uint8_t>(state, state_len));
    writer->extra = blake2b;
  }
  return writer;
}

void tachyon_halo2_bn254_transcript_writer_destroy(
    tachyon_halo2_bn254_transcript_writer* writer) {
  if (writer->type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
    Blake2bWriter* blake2b = reinterpret_cast<Blake2bWriter*>(writer->extra);
    delete blake2b;
  }
  delete writer;
}

void tachyon_halo2_bn254_transcript_writer_update(
    tachyon_halo2_bn254_transcript_writer* writer, const uint8_t* data,
    size_t data_len) {
  if (writer->type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
    Blake2bWriter* blake2b = reinterpret_cast<Blake2bWriter*>(writer->extra);
    blake2b->Update(data, data_len);
    return;
  }
  NOTREACHED();
}

void tachyon_halo2_bn254_transcript_writer_finalize(
    tachyon_halo2_bn254_transcript_writer* writer, uint8_t* data,
    size_t* data_len) {
  if (writer->type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
    *data_len = BLAKE2B512_DIGEST_LENGTH;
    if (data == nullptr) return;
    Blake2bWriter* blake2b = reinterpret_cast<Blake2bWriter*>(writer->extra);
    uint8_t data_tmp[BLAKE2B512_DIGEST_LENGTH];
    blake2b->Finalize(data_tmp);
    memcpy(data, data_tmp, BLAKE2B512_DIGEST_LENGTH);
    return;
  }
  NOTREACHED();
}

void tachyon_halo2_bn254_transcript_writer_get_state(
    const tachyon_halo2_bn254_transcript_writer* writer, uint8_t* state,
    size_t* state_len) {
  if (writer->type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
    *state_len = sizeof(blake2b_state_st);
    if (state == nullptr) return;
    Blake2bWriter* blake2b = reinterpret_cast<Blake2bWriter*>(writer->extra);
    std::vector<uint8_t> state_tmp = blake2b->GetState();
    memcpy(state, state_tmp.data(), sizeof(blake2b_state_st));
    return;
  }
  NOTREACHED();
}
