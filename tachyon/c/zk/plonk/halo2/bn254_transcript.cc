#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"

#include <string.h>

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/sha256_transcript.h"
#include "tachyon/zk/plonk/halo2/transcript_type.h"

using namespace tachyon;

using Blake2bWriter =
    zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>;
using PoseidonWriter =
    zk::plonk::halo2::PoseidonWriter<math::bn254::G1AffinePoint>;
using Sha256Writer = zk::plonk::halo2::Sha256Writer<math::bn254::G1AffinePoint>;

tachyon_halo2_bn254_transcript_writer*
tachyon_halo2_bn254_transcript_writer_create(uint8_t type) {
  tachyon_halo2_bn254_transcript_writer* writer =
      new tachyon_halo2_bn254_transcript_writer;
  writer->type = type;
  switch (static_cast<zk::plonk::halo2::TranscriptType>(type)) {
    case zk::plonk::halo2::TranscriptType::kBlake2b: {
      writer->extra = new Blake2bWriter(base::Uint8VectorBuffer());
      return writer;
    }
    case zk::plonk::halo2::TranscriptType::kPoseidon: {
      writer->extra = new PoseidonWriter(base::Uint8VectorBuffer());
      return writer;
    }
    case zk::plonk::halo2::TranscriptType::kSha256: {
      writer->extra = new Sha256Writer(base::Uint8VectorBuffer());
      return writer;
    }
  }
  NOTREACHED();
  return nullptr;
}

tachyon_halo2_bn254_transcript_writer*
tachyon_halo2_bn254_transcript_writer_create_from_state(uint8_t type,
                                                        const uint8_t* state,
                                                        size_t state_len) {
  tachyon_halo2_bn254_transcript_writer* writer =
      new tachyon_halo2_bn254_transcript_writer;
  writer->type = type;
  switch (static_cast<zk::plonk::halo2::TranscriptType>(type)) {
    case zk::plonk::halo2::TranscriptType::kBlake2b: {
      Blake2bWriter* blake2b = new Blake2bWriter(base::Uint8VectorBuffer());
      blake2b->SetState(absl::Span<const uint8_t>(state, state_len));
      writer->extra = blake2b;
      return writer;
    }
    case zk::plonk::halo2::TranscriptType::kPoseidon: {
      PoseidonWriter* poseidon = new PoseidonWriter(base::Uint8VectorBuffer());
      poseidon->SetState(absl::Span<const uint8_t>(state, state_len));
      writer->extra = poseidon;
      return writer;
    }
    case zk::plonk::halo2::TranscriptType::kSha256: {
      Sha256Writer* sha256 = new Sha256Writer(base::Uint8VectorBuffer());
      sha256->SetState(absl::Span<const uint8_t>(state, state_len));
      writer->extra = sha256;
      return writer;
    }
  }
  NOTREACHED();
  return nullptr;
}

void tachyon_halo2_bn254_transcript_writer_destroy(
    tachyon_halo2_bn254_transcript_writer* writer) {
  switch (static_cast<zk::plonk::halo2::TranscriptType>(writer->type)) {
    case zk::plonk::halo2::TranscriptType::kBlake2b: {
      delete reinterpret_cast<Blake2bWriter*>(writer->extra);
      delete writer;
      return;
    }
    case zk::plonk::halo2::TranscriptType::kPoseidon: {
      delete reinterpret_cast<PoseidonWriter*>(writer->extra);
      delete writer;
      return;
    }
    case zk::plonk::halo2::TranscriptType::kSha256: {
      delete reinterpret_cast<Sha256Writer*>(writer->extra);
      delete writer;
      return;
    }
  }
  NOTREACHED();
}

void tachyon_halo2_bn254_transcript_writer_update(
    tachyon_halo2_bn254_transcript_writer* writer, const uint8_t* data,
    size_t data_len) {
  switch (static_cast<zk::plonk::halo2::TranscriptType>(writer->type)) {
    case zk::plonk::halo2::TranscriptType::kBlake2b: {
      reinterpret_cast<Blake2bWriter*>(writer->extra)->Update(data, data_len);
      return;
    }
    case zk::plonk::halo2::TranscriptType::kPoseidon: {
      reinterpret_cast<PoseidonWriter*>(writer->extra)
          ->Update(reinterpret_cast<const PoseidonWriter::ScalarField*>(data),
                   data_len / PoseidonWriter::ScalarBigInt::kByteNums);
      return;
    }
    case zk::plonk::halo2::TranscriptType::kSha256: {
      reinterpret_cast<Sha256Writer*>(writer->extra)->Update(data, data_len);
      return;
    }
  }
  NOTREACHED();
}

void tachyon_halo2_bn254_transcript_writer_finalize(
    tachyon_halo2_bn254_transcript_writer* writer, uint8_t* data,
    size_t* data_len) {
  switch (static_cast<zk::plonk::halo2::TranscriptType>(writer->type)) {
    case zk::plonk::halo2::TranscriptType::kBlake2b: {
      *data_len = BLAKE2B512_DIGEST_LENGTH;
      if (data == nullptr) return;
      uint8_t data_tmp[BLAKE2B512_DIGEST_LENGTH];
      reinterpret_cast<Blake2bWriter*>(writer->extra)->Finalize(data_tmp);
      memcpy(data, data_tmp, BLAKE2B512_DIGEST_LENGTH);
      return;
    }
    case zk::plonk::halo2::TranscriptType::kPoseidon:
      break;
    case zk::plonk::halo2::TranscriptType::kSha256: {
      *data_len = SHA256_DIGEST_LENGTH;
      if (data == nullptr) return;
      uint8_t data_tmp[SHA256_DIGEST_LENGTH];
      reinterpret_cast<Sha256Writer*>(writer->extra)->Finalize(data_tmp);
      memcpy(data, data_tmp, SHA256_DIGEST_LENGTH);
      return;
    }
  }
  NOTREACHED();
}

tachyon_bn254_fr tachyon_halo2_bn254_transcript_writer_squeeze(
    tachyon_halo2_bn254_transcript_writer* writer) {
  tachyon_bn254_fr ret;
  math::bn254::Fr challenge;
  switch (static_cast<zk::plonk::halo2::TranscriptType>(writer->type)) {
    case zk::plonk::halo2::TranscriptType::kPoseidon: {
      challenge = reinterpret_cast<PoseidonWriter*>(writer->extra)->Squeeze();
      memcpy(ret.limbs, challenge.value().limbs,
             math::bn254::Fr::BigIntTy::kByteNums);
      return ret;
    }
    case zk::plonk::halo2::TranscriptType::kBlake2b:
    case zk::plonk::halo2::TranscriptType::kSha256:
      break;
  }
  NOTREACHED();
  return ret;
}

void tachyon_halo2_bn254_transcript_writer_get_state(
    const tachyon_halo2_bn254_transcript_writer* writer, uint8_t* state,
    size_t* state_len) {
  switch (static_cast<zk::plonk::halo2::TranscriptType>(writer->type)) {
    case zk::plonk::halo2::TranscriptType::kBlake2b: {
      Blake2bWriter* blake2b = reinterpret_cast<Blake2bWriter*>(writer->extra);
      *state_len = blake2b->GetStateLen();
      if (state == nullptr) return;
      std::vector<uint8_t> state_tmp = blake2b->GetState();
      memcpy(state, state_tmp.data(), *state_len);
      return;
    }
    case zk::plonk::halo2::TranscriptType::kPoseidon: {
      PoseidonWriter* poseidon =
          reinterpret_cast<PoseidonWriter*>(writer->extra);
      *state_len = poseidon->GetStateLen();
      if (state == nullptr) return;
      std::vector<uint8_t> state_tmp = poseidon->GetState();
      memcpy(state, state_tmp.data(), *state_len);
      return;
    }
    case zk::plonk::halo2::TranscriptType::kSha256: {
      Sha256Writer* sha256 = reinterpret_cast<Sha256Writer*>(writer->extra);
      *state_len = sha256->GetStateLen();
      if (state == nullptr) return;
      std::vector<uint8_t> state_tmp = sha256->GetState();
      memcpy(state, state_tmp.data(), *state_len);
      return;
    }
  }
  NOTREACHED();
}
