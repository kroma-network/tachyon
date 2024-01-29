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
namespace internal {

template <typename AffinePoint>
class Sha256Base {
 protected:
  using BaseField = typename AffinePoint::BaseField;
  using ScalarField = typename AffinePoint::ScalarField;

  Sha256Base() { SHA256_Init(&state_); }

  ScalarField DoSqueezeChallenge() {
    DoUpdate(kShaPrefixChallenge, 1);
    uint8_t result[SHA256_DIGEST_LENGTH] = {0};
    DoFinalize(result);

    DoInit();
    DoUpdate(result, SHA256_DIGEST_LENGTH);

    if constexpr (ScalarField::N <= 4) {
      return ScalarField::FromAnySizedBigInt(
          math::BigInt<4>::FromBytesLE(result));
    } else {
      base::AlwaysFalse<AffinePoint>();
    }
  }

  bool DoWriteToTranscript(const AffinePoint& point) {
    DoUpdate(kShaPrefixZeros, SHA256_DIGEST_LENGTH - 1);
    DoUpdate(kShaPrefixPoint, 1);
    DoUpdate(point.x().ToBigInt().ToBytesBE().data(),
             BaseField::BigIntTy::kByteNums);
    DoUpdate(point.y().ToBigInt().ToBytesBE().data(),
             BaseField::BigIntTy::kByteNums);
    return true;
  }

  bool DoWriteToTranscript(const ScalarField& scalar) {
    DoUpdate(kShaPrefixZeros, SHA256_DIGEST_LENGTH - 1);
    DoUpdate(kShaPrefixScalar, 1);
    DoUpdate(scalar.ToBigInt().ToBytesBE().data(),
             ScalarField::BigIntTy::kByteNums);
    return true;
  }

  void DoInit() { SHA256_Init(&state_); }

  void DoUpdate(const void* data, size_t len) {
    SHA256_Update(&state_, data, len);
  }

  void DoFinalize(uint8_t result[SHA256_DIGEST_LENGTH]) {
    SHA256_CTX hasher = state_;
    SHA256_Final(result, &hasher);
  }

  SHA256_CTX state_;
};

}  // namespace internal

template <typename AffinePoint>
class Sha256Reader : public crypto::TranscriptReader<AffinePoint>,
                     protected internal::Sha256Base<AffinePoint> {
 public:
  using ScalarField = typename AffinePoint::ScalarField;

  // Initialize a transcript given an input buffer.
  explicit Sha256Reader(base::Buffer read_buf)
      : crypto::TranscriptReader<AffinePoint>(std::move(read_buf)) {}

  void Init() { this->DoInit(); }

  void Update(const void* data, size_t len) { this->DoUpdate(data, len); }

  void Finalize(uint8_t result[SHA256_DIGEST_LENGTH]) {
    this->DoFinalize(result);
  }

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
class Sha256Writer : public crypto::TranscriptWriter<AffinePoint>,
                     protected internal::Sha256Base<AffinePoint> {
 public:
  using ScalarField = typename AffinePoint::ScalarField;

  // Initialize a transcript given an output buffer.
  explicit Sha256Writer(base::Uint8VectorBuffer write_buf)
      : crypto::TranscriptWriter<AffinePoint>(std::move(write_buf)) {
    SHA256_Init(&state_);
  }

  void Init() { this->DoInit(); }

  void Update(const void* data, size_t len) { this->DoUpdate(data, len); }

  void Finalize(uint8_t result[SHA256_DIGEST_LENGTH]) {
    this->DoFinalize(result);
  }

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

  SHA256_CTX state_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_SHA256_TRANSCRIPT_H_
