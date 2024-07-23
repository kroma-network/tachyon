#ifndef TACHYON_ZK_PLONK_HALO2_SNARK_VERIFIER_POSEIDON_TRANSCRIPT_H_
#define TACHYON_ZK_PLONK_HALO2_SNARK_VERIFIER_POSEIDON_TRANSCRIPT_H_

#include <string.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/zk/plonk/halo2/prime_field_conversion.h"
#include "tachyon/zk/plonk/halo2/proof_serializer.h"

namespace tachyon::zk::plonk::halo2 {
namespace internal {

template <typename AffinePoint>
class SnarkVerifierPoseidonBase {
 protected:
  using BaseField = typename AffinePoint::BaseField;
  using ScalarField = typename AffinePoint::ScalarField;
  using Curve = typename AffinePoint::Curve;
  using CurveConfig = typename Curve::Config;

  SnarkVerifierPoseidonBase()
      : poseidon_(
            // See
            // https://github.com/scroll-tech/snark-verifier/blob/58c46b7/snark-verifier-sdk/src/param.rs#L7-L10.
            crypto::PoseidonConfig<ScalarField>::CreateCustom(4, 5, 8, 60, 0)),
        state_(poseidon_.config) {
    // See
    // https://github.com/scroll-tech/snark-verifier/blob/58c46b7/snark-verifier/src/util/hash/poseidon.rs#L28-L31.
    state_.elements[0] = FromUint128<ScalarField>(absl::uint128(1) << 64);
  }

  ScalarField DoSqueezeChallenge() {
    ScalarField scalar = DoSqueeze();
    std::array<uint8_t, 32> scalar_bytes = scalar.ToBigInt().ToBytesLE();
    uint8_t wide_scalar_bytes[64] = {0};
    memcpy(wide_scalar_bytes, scalar_bytes.data(), 32);
    return FromUint512<ScalarField>(wide_scalar_bytes);
  }

  bool DoWriteToTranscript(const AffinePoint& point) {
    ScalarField coords[] = {BaseToScalar(point.x()), BaseToScalar(point.y())};
    DoUpdate(coords, 2);
    return true;
  }

  bool DoWriteToTranscript(const ScalarField& scalar) {
    DoUpdate(&scalar, 1);
    return true;
  }

  ScalarField DoSqueeze() {
    size_t rate = poseidon_.config.rate;
    size_t num_chunks = (buf_.size() + rate - 1) / rate;

    for (size_t i = 0; i < num_chunks; ++i) {
      size_t start = i * rate;
      size_t len = std::min(start + rate, buf_.size()) - start;
      // See
      // https://github.com/scroll-tech/snark-verifier/blob/58c46b7/snark-verifier/src/util/hash/poseidon.rs#L57-L61.
      for (size_t j = 0; j < len; ++j) {
        state_[j + 1] += buf_[start + j];
      }
      // See
      // https://github.com/scroll-tech/snark-verifier/blob/58c46b7/snark-verifier/src/util/hash/poseidon.rs#L70-L72.
      if (len + 1 < state_.size()) {
        state_[len + 1] += ScalarField::One();
      }
      poseidon_.Permute(state_);
    }

    if (buf_.size() == num_chunks * rate) {
      // See
      // https://github.com/scroll-tech/snark-verifier/blob/58c46b7/snark-verifier/src/util/hash/poseidon.rs#L70-L72.
      state_[1] += ScalarField::One();
      poseidon_.Permute(state_);
    }

    buf_.clear();
    return state_[1];
  }

  void DoUpdate(const ScalarField* data, size_t len) {
    buf_.reserve(buf_.size() + len);
    for (size_t i = 0; i < len; ++i) {
      buf_.push_back(data[i]);
    }
  }

  std::vector<uint8_t> DoGetState() const {
    base::Uint8VectorBuffer buffer;
    buffer.set_endian(base::Endian::kLittle);
    CHECK(buffer.Grow(base::EstimateSize(buf_, state_)));
    CHECK(buffer.WriteMany(buf_, state_));
    CHECK(buffer.Done());
    return std::move(buffer).TakeOwnedBuffer();
  }

  void DoSetState(absl::Span<const uint8_t> state) {
    base::ReadOnlyBuffer buffer(state.data(), state.size());
    buffer.set_endian(base::Endian::kLittle);
    CHECK(buffer.ReadMany(&buf_, &state_));
    CHECK(buffer.Done());
  }

  crypto::PoseidonSponge<ScalarField> poseidon_;
  std::vector<ScalarField> buf_;
  crypto::SpongeState<ScalarField> state_;

 private:
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a369/halo2_proofs/src/helpers.rs#L37-L58.
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
class SnarkVerifierPoseidonReader
    : public crypto::TranscriptReader<AffinePoint>,
      protected internal::SnarkVerifierPoseidonBase<AffinePoint> {
 public:
  using ScalarField = typename AffinePoint::ScalarField;

  // Initialize a transcript given an input buffer.
  explicit SnarkVerifierPoseidonReader(base::ReadOnlyBuffer buffer)
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
class SnarkVerifierPoseidonWriter
    : public crypto::TranscriptWriter<AffinePoint>,
      protected internal::SnarkVerifierPoseidonBase<AffinePoint> {
 public:
  using ScalarField = typename AffinePoint::ScalarField;
  using ScalarBigInt = typename ScalarField::BigIntTy;

  // Initialize a transcript given an output buffer.
  explicit SnarkVerifierPoseidonWriter(base::Uint8VectorBuffer buffer)
      : crypto::TranscriptWriter<AffinePoint>(std::move(buffer)) {}

  // NOTE(chokobole): |GetDigestLen()|, |GetStateLen()|, |Update()|,
  // |Squeeze()|, |GetState()| and |SetState()| are called from rust binding.
  size_t GetDigestLen() const { return ScalarBigInt::kByteNums; }

  size_t GetStateLen() const {
    return base::EstimateSize(this->state_, this->buf_);
  }

  void Update(const ScalarField* data, size_t len) {
    this->DoUpdate(data, len);
  }

  ScalarField Squeeze() { return this->DoSqueeze(); }

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

#endif  // TACHYON_ZK_PLONK_HALO2_SNARK_VERIFIER_POSEIDON_TRANSCRIPT_H_
