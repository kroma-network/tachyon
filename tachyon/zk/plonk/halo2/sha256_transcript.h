// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_SHA256_TRANSCRIPT_H_
#define TACHYON_ZK_PLONK_HALO2_SHA256_TRANSCRIPT_H_

#include <utility>

#include "openssl/sha.h"

#include "tachyon/base/types/always_false.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/halo2/proof_serializer.h"

namespace tachyon::zk::halo2 {

template <typename AffinePointTy>
class Sha256Reader : public crypto::TranscriptReader<AffinePointTy> {
 public:
  using BaseField = typename AffinePointTy::BaseField;
  using ScalarField = typename AffinePointTy::ScalarField;

  Sha256Reader() = default;
  // Initialize a transcript given an input buffer.
  explicit Sha256Reader(base::Buffer read_buf)
      : crypto::TranscriptReader<AffinePointTy>(std::move(read_buf)) {
    SHA256_Init(&state_);
  }

  // crypto::TranscriptReader methods
  ScalarField SqueezeChallenge() override {
    SHA256_Update(&state_, kShaPrefixChallenge, 1);
    SHA256_CTX hasher = state_;
    uint8_t result[32] = {0};
    SHA256_Final(result, &hasher);

    SHA256_Init(&state_);
    SHA256_Update(&state_, result, 32);

    if constexpr (ScalarField::N <= 4) {
      return ScalarField::FromAnySizedBigInt(
          math::BigInt<4>::FromBytesLE(result));
    } else {
      base::AlwaysFalse<AffinePointTy>();
    }
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    SHA256_Update(&state_, kShaPrefixZeros, 31);
    SHA256_Update(&state_, kShaPrefixPoint, 1);
    SHA256_Update(&state_, point.x().ToBigInt().ToBytesBE().data(),
                  BaseField::BigIntTy::kByteNums);
    SHA256_Update(&state_, point.y().ToBigInt().ToBytesBE().data(),
                  BaseField::BigIntTy::kByteNums);
    return true;
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    SHA256_Update(&state_, kShaPrefixZeros, 31);
    SHA256_Update(&state_, kShaPrefixScalar, 1);
    SHA256_Update(&state_, scalar.ToBigInt().ToBytesBE().data(),
                  ScalarField::BigIntTy::kByteNums);
    return true;
  }

 private:
  bool DoReadFromProof(AffinePointTy* point) const override {
    return ProofSerializer<AffinePointTy>::ReadFromProof(this->buffer_, point);
  }

  bool DoReadFromProof(ScalarField* scalar) const override {
    return ProofSerializer<ScalarField>::ReadFromProof(this->buffer_, scalar);
  }

  SHA256_CTX state_;
};

template <typename AffinePointTy>
class Sha256Writer : public crypto::TranscriptWriter<AffinePointTy> {
 public:
  using BaseField = typename AffinePointTy::BaseField;
  using ScalarField = typename AffinePointTy::ScalarField;

  Sha256Writer() = default;
  // Initialize a transcript given an output buffer.
  explicit Sha256Writer(base::Uint8VectorBuffer write_buf)
      : crypto::TranscriptWriter<AffinePointTy>(std::move(write_buf)) {
    SHA256_Init(&state_);
  }

  // crypto::TranscriptWriter methods
  ScalarField SqueezeChallenge() override {
    SHA256_Update(&state_, kShaPrefixChallenge, 1);
    SHA256_CTX hasher = state_;
    uint8_t result[32] = {0};
    SHA256_Final(result, &hasher);

    SHA256_Init(&state_);
    SHA256_Update(&state_, result, 32);

    if constexpr (ScalarField::N <= 4) {
      return ScalarField::FromAnySizedBigInt(
          math::BigInt<4>::FromBytesLE(result));
    } else {
      base::AlwaysFalse<AffinePointTy>();
    }
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    SHA256_Update(&state_, kShaPrefixZeros, 31);
    SHA256_Update(&state_, kShaPrefixPoint, 1);
    SHA256_Update(&state_, point.x().ToBigInt().ToBytesBE().data(),
                  BaseField::BigIntTy::kByteNums);
    SHA256_Update(&state_, point.y().ToBigInt().ToBytesBE().data(),
                  BaseField::BigIntTy::kByteNums);
    return true;
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    SHA256_Update(&state_, kShaPrefixZeros, 31);
    SHA256_Update(&state_, kShaPrefixScalar, 1);
    SHA256_Update(&state_, scalar.ToBigInt().ToBytesBE().data(),
                  ScalarField::BigIntTy::kByteNums);
    return true;
  }

 private:
  bool DoWriteToProof(const AffinePointTy& point) override {
    return ProofSerializer<AffinePointTy>::WriteToProof(point, this->buffer_);
  }

  bool DoWriteToProof(const ScalarField& scalar) override {
    return ProofSerializer<ScalarField>::WriteToProof(scalar, this->buffer_);
  }

  SHA256_CTX state_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_SHA256_TRANSCRIPT_H_
