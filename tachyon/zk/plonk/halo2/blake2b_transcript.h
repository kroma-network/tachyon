// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_BLAKE2B_TRANSCRIPT_H_
#define TACHYON_ZK_PLONK_HALO2_BLAKE2B_TRANSCRIPT_H_

#include <stdint.h>

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "openssl/blake2.h"

#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/halo2/proof_serializer.h"

namespace tachyon::zk::plonk::halo2 {
namespace internal {

template <typename AffinePoint>
class Blake2bBase {
 protected:
  using BaseField = typename AffinePoint::BaseField;
  using ScalarField = typename AffinePoint::ScalarField;

  Blake2bBase() { BLAKE2B512_InitWithPersonal(&state_, kTranscriptStr); }

  ScalarField DoSqueezeChallenge() {
    DoUpdate(kBlake2bPrefixChallenge, 1);
    uint8_t result[BLAKE2B512_DIGEST_LENGTH] = {0};
    DoFinalize(result);
    return ScalarField::FromAnySizedBigInt(
        math::BigInt<8>::FromBytesLE(result));
  }

  bool DoWriteToTranscript(const AffinePoint& point) {
    DoUpdate(kBlake2bPrefixPoint, 1);
    if (point.infinity()) {
      DoUpdate(BaseField::BigIntTy::Zero().ToBytesLE().data(),
               BaseField::BigIntTy::kByteNums);
      DoUpdate(typename BaseField::BigIntTy(5).ToBytesLE().data(),
               BaseField::BigIntTy::kByteNums);
    } else {
      DoUpdate(point.x().ToBigInt().ToBytesLE().data(),
               BaseField::BigIntTy::kByteNums);
      DoUpdate(point.y().ToBigInt().ToBytesLE().data(),
               BaseField::BigIntTy::kByteNums);
    }
    return true;
  }

  bool DoWriteToTranscript(const ScalarField& scalar) {
    DoUpdate(kBlake2bPrefixScalar, 1);
    DoUpdate(scalar.ToBigInt().ToBytesLE().data(),
             ScalarField::BigIntTy::kByteNums);
    return true;
  }

  void DoUpdate(const void* data, size_t len) {
    BLAKE2B512_Update(&state_, data, len);
  }

  void DoFinalize(uint8_t result[BLAKE2B512_DIGEST_LENGTH]) {
    BLAKE2B_CTX hasher = state_;
    BLAKE2B512_Final(result, &hasher);
  }

  std::vector<uint8_t> DoGetState() const {
    const blake2b_state_st* state_impl =
        reinterpret_cast<const blake2b_state_st*>(&state_);
    base::Uint8VectorBuffer buffer;
    buffer.set_endian(base::Endian::kLittle);
    CHECK(buffer.Grow(sizeof(blake2b_state_st)));
    CHECK(buffer.Write(state_impl->h));
    CHECK(buffer.Write(state_impl->t_low));
    CHECK(buffer.Write(state_impl->t_high));
    CHECK(buffer.Write(state_impl->block));
    CHECK(buffer.Write(state_impl->block_used));
    return std::move(buffer).TakeOwnedBuffer();
  }

  void DoSetState(absl::Span<const uint8_t> state) {
    base::Buffer buffer(const_cast<uint8_t*>(state.data()), state.size());
    buffer.set_endian(base::Endian::kLittle);
    blake2b_state_st* state_impl = reinterpret_cast<blake2b_state_st*>(&state_);
    CHECK(buffer.Read(state_impl->h));
    CHECK(buffer.Read(&state_impl->t_low));
    CHECK(buffer.Read(&state_impl->t_high));
    CHECK(buffer.Read(state_impl->block));
    CHECK(buffer.Read(&state_impl->block_used));
  }

  BLAKE2B_CTX state_;
};

}  // namespace internal

// TODO(TomTaehoonKim): We will replace Blake2b with an algebraic hash function
// in a later version. See
// https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/transcript/blake2b.rs#L25
template <typename AffinePoint>
class Blake2bReader : public crypto::TranscriptReader<AffinePoint>,
                      protected internal::Blake2bBase<AffinePoint> {
 public:
  using ScalarField = typename AffinePoint::ScalarField;

  // Initialize a transcript given an input buffer.
  explicit Blake2bReader(base::Buffer read_buf)
      : crypto::TranscriptReader<AffinePoint>(std::move(read_buf)) {}

  // crypto::TranscriptReader methods
  ScalarField SqueezeChallenge() override { return this->DoSqueezeChallenge(); }

  bool WriteToTranscript(const AffinePoint& point) override {
    return this->DoWriteToTranscript(point);
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    return this->DoWriteToTranscript(scalar);
  }

 private:
  bool DoReadFromProof(AffinePoint* point) const override {
    return ProofSerializer<AffinePoint>::ReadFromProof(this->buffer_, point);
  }

  bool DoReadFromProof(ScalarField* scalar) const override {
    return ProofSerializer<ScalarField>::ReadFromProof(this->buffer_, scalar);
  }
};

template <typename AffinePoint>
class Blake2bWriter : public crypto::TranscriptWriter<AffinePoint>,
                      protected internal::Blake2bBase<AffinePoint> {
 public:
  using ScalarField = typename AffinePoint::ScalarField;

  // Initialize a transcript given an output buffer.
  explicit Blake2bWriter(base::Uint8VectorBuffer write_buf)
      : crypto::TranscriptWriter<AffinePoint>(std::move(write_buf)) {}

  // NOTE(chokobole): |Update()|, |Finalize()|, |GetState()| and |SetState()|
  // are called from rust binding.
  void Update(const void* data, size_t len) { this->DoUpdate(data, len); }

  void Finalize(uint8_t result[BLAKE2B512_DIGEST_LENGTH]) {
    this->DoFinalize(result);
  }

  std::vector<uint8_t> GetState() const { return this->DoGetState(); }

  void SetState(absl::Span<const uint8_t> state) { this->DoSetState(state); }

  // crypto::TranscriptWriter methods
  ScalarField SqueezeChallenge() override { return this->DoSqueezeChallenge(); }

  bool WriteToTranscript(const AffinePoint& point) override {
    return this->DoWriteToTranscript(point);
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    return this->DoWriteToTranscript(scalar);
  }

 private:
  bool DoWriteToProof(const AffinePoint& point) override {
    return ProofSerializer<AffinePoint>::WriteToProof(point, this->buffer_);
  }

  bool DoWriteToProof(const ScalarField& scalar) override {
    return ProofSerializer<ScalarField>::WriteToProof(scalar, this->buffer_);
  }
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_BLAKE2B_TRANSCRIPT_H_
