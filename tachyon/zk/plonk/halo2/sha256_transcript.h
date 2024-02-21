// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_SHA256_TRANSCRIPT_H_
#define TACHYON_ZK_PLONK_HALO2_SHA256_TRANSCRIPT_H_

#include <utility>
#include <vector>

#include "openssl/sha.h"

#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/halo2/prime_field_conversion.h"
#include "tachyon/zk/plonk/halo2/proof_serializer.h"

namespace tachyon::zk::plonk::halo2 {
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

    SHA256_Init(&state_);
    DoUpdate(result, SHA256_DIGEST_LENGTH);

    uint8_t expanded_result[SHA256_DIGEST_LENGTH * 2] = {0};
    memcpy(expanded_result, result, SHA256_DIGEST_LENGTH);
    return FromUint512<ScalarField>(expanded_result);
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

  void DoUpdate(const void* data, size_t len) {
    SHA256_Update(&state_, data, len);
  }

  void DoFinalize(uint8_t result[SHA256_DIGEST_LENGTH]) {
    SHA256_CTX hasher = state_;
    SHA256_Final(result, &hasher);
  }

  std::vector<uint8_t> DoGetState() const {
    const sha256_state_st* state_impl =
        reinterpret_cast<const sha256_state_st*>(&state_);
    base::Uint8VectorBuffer buffer;
    buffer.set_endian(base::Endian::kLittle);
    CHECK(buffer.Grow(sizeof(sha256_state_st)));
    CHECK(buffer.WriteMany(state_impl->h, state_impl->Nl, state_impl->Nh,
                           state_impl->data, state_impl->num,
                           state_impl->md_len));
    CHECK(buffer.Done());
    return std::move(buffer).TakeOwnedBuffer();
  }

  void DoSetState(absl::Span<const uint8_t> state) {
    base::ReadOnlyBuffer buffer(state.data(), state.size());
    buffer.set_endian(base::Endian::kLittle);
    sha256_state_st* state_impl = reinterpret_cast<sha256_state_st*>(&state_);
    CHECK(buffer.ReadMany(state_impl->h, &state_impl->Nl, &state_impl->Nh,
                          state_impl->data, &state_impl->num,
                          &state_impl->md_len));
    CHECK(buffer.Done());
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
  explicit Sha256Reader(base::ReadOnlyBuffer read_buf)
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
    using BaseField = typename AffinePoint::BaseField;

    // See
    // https://github.com/kroma-network/halo2-snark-aggregator/blob/2637b512397b255525782006439a9cedde5b79b8/halo2-snark-aggregator-api/src/transcript/sha.rs#L43-L61.
    math::BigInt<4> x;
    math::BigInt<4> y;
    if (!this->buffer_.ReadMany(&x, &y)) return false;
    *point = {BaseField(std::move(x)), BaseField(std::move(y))};
    return true;
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

  // NOTE(chokobole): |GetDigestLen()|, |GetStateLen()|, |Update()|,
  // |Finalize()|, |GetState()| and |SetState()| are called from rust binding.
  size_t GetDigestLen() const { return SHA256_DIGEST_LENGTH; }

  size_t GetStateLen() const { return sizeof(sha256_state_st); }

  void Update(const void* data, size_t len) { this->DoUpdate(data, len); }

  void Finalize(uint8_t result[SHA256_DIGEST_LENGTH]) {
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
    // See
    // https://github.com/kroma-network/halo2-snark-aggregator/blob/2637b512397b255525782006439a9cedde5b79b8/halo2-snark-aggregator-api/src/transcript/sha.rs#L156-L173.
    math::BigInt<4> x = point.x().ToBigInt();
    math::BigInt<4> y = point.y().ToBigInt();
    return this->buffer_.WriteMany(x, y);
  }

  bool DoWriteToProof(const ScalarField& scalar) override {
    return ProofSerializer<ScalarField>::WriteToProof(scalar, this->buffer_);
  }

  SHA256_CTX state_;
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_SHA256_TRANSCRIPT_H_
