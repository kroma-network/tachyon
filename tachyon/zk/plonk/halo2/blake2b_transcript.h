// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_BLAKE2B_TRANSCRIPT_H_
#define TACHYON_ZK_PLONK_HALO2_BLAKE2B_TRANSCRIPT_H_

#include <utility>

#include "openssl/blake2.h"

#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/halo2/proof_serializer.h"

namespace tachyon::zk::halo2 {
namespace internal {

template <typename AffinePointTy>
class Blake2bBase {
 protected:
  using BaseField = typename AffinePointTy::BaseField;
  using ScalarField = typename AffinePointTy::ScalarField;

  Blake2bBase() { BLAKE2B512_InitWithPersonal(&state_, kTranscriptStr); }

  ScalarField DoSqueezeChallenge() {
    BLAKE2B512_Update(&state_, kBlake2bPrefixChallenge, 1);
    BLAKE2B_CTX hasher = state_;
    uint8_t result[64] = {0};
    BLAKE2B512_Final(result, &hasher);
    return ScalarField::FromAnySizedBigInt(
        math::BigInt<8>::FromBytesLE(result));
  }

  bool DoWriteToTranscript(const AffinePointTy& point) {
    BLAKE2B512_Update(&state_, kBlake2bPrefixPoint, 1);
    if (point.infinity()) {
      BLAKE2B512_Update(&state_, BaseField::BigIntTy::Zero().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
      BLAKE2B512_Update(&state_,
                        typename BaseField::BigIntTy(5).ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
    } else {
      BLAKE2B512_Update(&state_, point.x().ToBigInt().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
      BLAKE2B512_Update(&state_, point.y().ToBigInt().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
    }
    return true;
  }

  bool DoWriteToTranscript(const ScalarField& scalar) {
    BLAKE2B512_Update(&state_, kBlake2bPrefixScalar, 1);
    BLAKE2B512_Update(&state_, scalar.ToBigInt().ToBytesLE().data(),
                      ScalarField::BigIntTy::kByteNums);
    return true;
  }

  BLAKE2B_CTX state_;
};

}  // namespace internal

// TODO(TomTaehoonKim): We will replace Blake2b with an algebraic hash function
// in a later version. See
// https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/transcript/blake2b.rs#L25
template <typename AffinePointTy>
class Blake2bReader : public crypto::TranscriptReader<AffinePointTy>,
                      protected internal::Blake2bBase<AffinePointTy> {
 public:
  using ScalarField = typename AffinePointTy::ScalarField;

  // Initialize a transcript given an input buffer.
  explicit Blake2bReader(base::Buffer read_buf)
      : crypto::TranscriptReader<AffinePointTy>(std::move(read_buf)) {}

  // crypto::TranscriptReader methods
  ScalarField SqueezeChallenge() override { return this->DoSqueezeChallenge(); }

  bool WriteToTranscript(const AffinePointTy& point) override {
    return this->DoWriteToTranscript(point);
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    return this->DoWriteToTranscript(scalar);
  }

 private:
  bool DoReadFromProof(AffinePointTy* point) const override {
    return ProofSerializer<AffinePointTy>::ReadFromProof(this->buffer_, point);
  }

  bool DoReadFromProof(ScalarField* scalar) const override {
    return ProofSerializer<ScalarField>::ReadFromProof(this->buffer_, scalar);
  }
};

template <typename AffinePointTy>
class Blake2bWriter : public crypto::TranscriptWriter<AffinePointTy>,
                      protected internal::Blake2bBase<AffinePointTy> {
 public:
  using ScalarField = typename AffinePointTy::ScalarField;

  // Initialize a transcript given an output buffer.
  explicit Blake2bWriter(base::Uint8VectorBuffer write_buf)
      : crypto::TranscriptWriter<AffinePointTy>(std::move(write_buf)) {}

  // crypto::TranscriptWriter methods
  ScalarField SqueezeChallenge() override { return this->DoSqueezeChallenge(); }

  bool WriteToTranscript(const AffinePointTy& point) override {
    return this->DoWriteToTranscript(point);
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    return this->DoWriteToTranscript(scalar);
  }

 private:
  bool DoWriteToProof(const AffinePointTy& point) override {
    return ProofSerializer<AffinePointTy>::WriteToProof(point, this->buffer_);
  }

  bool DoWriteToProof(const ScalarField& scalar) override {
    return ProofSerializer<ScalarField>::WriteToProof(scalar, this->buffer_);
  }
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_BLAKE2B_TRANSCRIPT_H_
