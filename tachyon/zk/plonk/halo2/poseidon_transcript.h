// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_
#define TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/zk/plonk/halo2/poseidon_sponge.h"
#include "tachyon/zk/plonk/halo2/prime_field_conversion.h"
#include "tachyon/zk/plonk/halo2/proof_serializer.h"

namespace tachyon::zk::plonk::halo2 {
namespace internal {

template <typename AffinePoint>
class PoseidonBase {
 protected:
  using BaseField = typename AffinePoint::BaseField;
  using ScalarField = typename AffinePoint::ScalarField;
  using Curve = typename AffinePoint::Curve;
  using CurveConfig = typename Curve::Config;

  PoseidonBase()
      : state_(crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63,
                                                                 0)) {}

  ScalarField DoSqueezeChallenge() {
    return state_.SqueezeNativeFieldElements(1)[0];
  }

  bool DoWriteToTranscript(const AffinePoint& point) {
    std::array<ScalarField, 2> coords = {BaseToScalar(point.x()),
                                         BaseToScalar(point.y())};
    return state_.Absorb(coords);
  }

  bool DoWriteToTranscript(const ScalarField& scalar) {
    return state_.Absorb(scalar);
  }

  void DoUpdate(const ScalarField* data, size_t len) {
    CHECK(state_.Absorb(absl::Span<const ScalarField>(data, len)));
  }

  std::vector<uint8_t> DoGetState() const {
    base::Uint8VectorBuffer buffer;
    buffer.set_endian(base::Endian::kLittle);
    CHECK(buffer.Grow(base::EstimateSize(state_.state)));
    CHECK(buffer.Write(state_.state));
    CHECK(buffer.Done());
    return std::move(buffer).TakeOwnedBuffer();
  }

  void DoSetState(absl::Span<const uint8_t> state) {
    base::ReadOnlyBuffer buffer(state.data(), state.size());
    buffer.set_endian(base::Endian::kLittle);
    CHECK(buffer.Read(&state_.state));
    CHECK(buffer.Done());
  }

  PoseidonSponge<ScalarField> state_;

 private:
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/helpers.rs#L37-L58.
  ScalarField BaseToScalar(const BaseField& base) {
    constexpr size_t kByteNums = BaseField::BigIntTy::kByteNums;

    std::array<uint8_t, kByteNums> bytes = base.ToBigInt().ToBytesLE();
    uint64_t buf[64] = {0};
    memcpy(buf, bytes.data(), std::min(size_t{64}, kByteNums));
    return FromUint512<ScalarField>(buf);
  }
};

}  // namespace internal

template <typename AffinePoint>
class PoseidonReader : public crypto::TranscriptReader<AffinePoint>,
                       protected internal::PoseidonBase<AffinePoint> {
 public:
  using ScalarField = typename AffinePoint::ScalarField;

  // Initialize a transcript given an input buffer.
  explicit PoseidonReader(base::ReadOnlyBuffer buffer)
      : crypto::TranscriptReader<AffinePoint>(std::move(buffer)) {}

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
class PoseidonWriter : public crypto::TranscriptWriter<AffinePoint>,
                       protected internal::PoseidonBase<AffinePoint> {
 public:
  using ScalarField = typename AffinePoint::ScalarField;
  using ScalarBigInt = typename ScalarField::BigIntTy;

  // Initialize a transcript given an output buffer.
  explicit PoseidonWriter(base::Uint8VectorBuffer buffer)
      : crypto::TranscriptWriter<AffinePoint>(std::move(buffer)) {}

  // NOTE(chokobole): |GetDigestLen()|, |GetStateLen()|, |Update()|,
  // |Squeeze()|, |GetState()| and |SetState()| are called from rust binding.
  size_t GetDigestLen() const { return ScalarBigInt::kByteNums; }

  size_t GetStateLen() const { return base::EstimateSize(this->state_.state); }

  void Update(const ScalarField* data, size_t len) {
    this->DoUpdate(data, len);
  }

  ScalarField Squeeze() {
    return this->state_.SqueezeNativeFieldElements(1)[0];
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

#endif  // TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_
