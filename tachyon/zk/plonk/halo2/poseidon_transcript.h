// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_
#define TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_

#include <string.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/strings/string_util.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"
#include "tachyon/crypto/transcripts/transcript.h"
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
      : poseidon_(
            // See
            // https://github.com/kroma-network/halo2/blob/7d0a369/halo2_proofs/src/transcript/poseidon.rs#L28.
            crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63, 0)),
        state_(poseidon_.config) {
    // See
    // https://github.com/kroma-network/poseidon/blob/00a2fe0/src/spec.rs#L15.
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

  // See
  // https://github.com/kroma-network/poseidon/blob/00a2fe0/src/poseidon.rs#L47-L69
  ScalarField DoSqueeze() {
    std::vector<ScalarField> last_chunk = absorbing_;

    // Add the finishing sign of the variable length hashing. Note that this mut
    // is also applied when absorbing line is empty.
    last_chunk.push_back(ScalarField::One());

    // Add the last chunk of inputs to the state for the final permutation
    // cycle.
    for (size_t i = 0; i < last_chunk.size(); ++i) {
      state_[i + 1] += last_chunk[i];
    }

    // Perform final permutation.
    poseidon_.Permute(state_);

    // Flush the absorption line.
    absorbing_.clear();
    return state_[1];
  }

  // See
  // https://github.com/kroma-network/poseidon/blob/00a2fe0/src/poseidon.rs#L23-L45.
  void DoUpdate(const ScalarField* data, size_t len) {
    std::vector<ScalarField> input_elements = absorbing_;
    input_elements.insert(input_elements.end(), data, data + len);

    size_t num_chunks = (input_elements.size() + poseidon_.config.rate - 1) /
                        poseidon_.config.rate;
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t start = i * poseidon_.config.rate;
      size_t chunk_len = i == num_chunks - 1 ? input_elements.size() - start
                                             : poseidon_.config.rate;
      absl::Span<const ScalarField> chunk(input_elements.data() + start,
                                          chunk_len);
      if (chunk_len < poseidon_.config.rate) {
        absorbing_ = std::vector<ScalarField>(chunk.begin(), chunk.end());
      } else {
        // Add new chunk of inputs for the next permutation cycle.
        for (size_t i = 0; i < poseidon_.config.rate; ++i) {
          state_[i + 1] += chunk[i];
        }

        // Perform intermediate permutation.
        poseidon_.Permute(state_);

        // Flush the absorption line.
        absorbing_.clear();
      }
    }
  }

  std::vector<uint8_t> DoGetState() const {
    base::Uint8VectorBuffer buffer;
    buffer.set_endian(base::Endian::kLittle);
    CHECK(buffer.Grow(base::EstimateSize(state_, absorbing_)));
    CHECK(buffer.WriteMany(state_, absorbing_));
    CHECK(buffer.Done());
    return std::move(buffer).TakeOwnedBuffer();
  }

  void DoSetState(absl::Span<const uint8_t> state) {
    base::ReadOnlyBuffer buffer(state.data(), state.size());
    buffer.set_endian(base::Endian::kLittle);
    CHECK(buffer.ReadMany(&state_, &absorbing_));
    CHECK(buffer.Done());
  }

  crypto::PoseidonSponge<ScalarField> poseidon_;
  std::vector<ScalarField> absorbing_;
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

  size_t GetStateLen() const {
    return base::EstimateSize(this->state_, this->absorbing_);
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

#endif  // TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_
