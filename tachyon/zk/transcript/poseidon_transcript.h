// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_TRANSCRIPT_POSEIDON_TRANSCRIPT_H_
#define TACHYON_ZK_TRANSCRIPT_POSEIDON_TRANSCRIPT_H_

#include <array>
#include <utility>

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/hashes/sponge/poseidon/halo2_poseidon.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/zk/transcript/transcript.h"

namespace tachyon::zk {

template <typename Curve>
class PoseidonReader : public TranscriptReader<math::AffinePoint<Curve>> {
 public:
  using AffinePointTy = typename Curve::AffinePointTy;
  using ScalarField = typename Curve::ScalarField;
  using CurveConfig = typename Curve::Config;

  PoseidonReader() = default;
  // Initialize a transcript given an input buffer.
  explicit PoseidonReader(base::Buffer buffer)
      : state_(
            crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63, 0)),
        buffer_(std::move(buffer)) {}

  base::Buffer& buffer() { return buffer_; }
  const base::Buffer& buffer() const { return buffer_; }

  // Transcript methods
  Challenge255<AffinePointTy> SqueezeChallenge() override {
    return Challenge255<AffinePointTy>(state_.SqueezeNativeFieldElements(1)[0]);
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    std::array<ScalarField, 2> coords = {CurveConfig::BaseToScalar(point.x()),
                                         CurveConfig::BaseToScalar(point.y())};
    return state_.Absorb(coords);
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    return state_.Absorb(scalar);
  }

  // TranscriptRead methods
  bool ReadPoint(AffinePointTy* point) override {
    return buffer_.Read(point) && WriteToTranscript(*point);
  }

  bool ReadScalar(ScalarField* scalar) override {
    return buffer_.Read(scalar) && WriteToTranscript(*scalar);
  }

 private:
  crypto::Halo2PoseidonSponge<ScalarField> state_;
  base::Buffer buffer_;
};

template <typename Curve>
class PoseidonWriter : public TranscriptWriter<math::AffinePoint<Curve>> {
 public:
  using AffinePointTy = typename Curve::AffinePointTy;
  using ScalarField = typename Curve::ScalarField;
  using CurveConfig = typename Curve::Config;

  PoseidonWriter() = default;
  // Initialize a transcript given an output buffer.
  explicit PoseidonWriter(base::VectorBuffer buffer)
      : state_(
            crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63, 0)),
        buffer_(std::move(buffer)) {}

  base::VectorBuffer& buffer() { return buffer_; }
  const base::VectorBuffer& buffer() const { return buffer_; }

  // Transcript methods
  Challenge255<AffinePointTy> SqueezeChallenge() override {
    return Challenge255<AffinePointTy>(state_.SqueezeNativeFieldElements(1)[0]);
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    std::array<ScalarField, 2> coords = {CurveConfig::BaseToScalar(point.x()),
                                         CurveConfig::BaseToScalar(point.y())};
    return state_.Absorb(coords);
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    return state_.Absorb(scalar);
  }

  // TranscriptWrite methods
  bool WriteToProof(const AffinePointTy& point) override {
    return WriteToTranscript(point) && buffer_.Write(point);
  }

  bool WriteToProof(const ScalarField& scalar) override {
    return WriteToTranscript(scalar) && buffer_.Write(scalar);
  }

 private:
  crypto::Halo2PoseidonSponge<ScalarField> state_;
  base::VectorBuffer buffer_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_TRANSCRIPT_POSEIDON_TRANSCRIPT_H_
