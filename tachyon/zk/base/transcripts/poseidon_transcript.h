// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_TRANSCRIPTS_POSEIDON_TRANSCRIPT_H_
#define TACHYON_ZK_BASE_TRANSCRIPTS_POSEIDON_TRANSCRIPT_H_

#include <array>
#include <utility>

#include "tachyon/crypto/hashes/sponge/poseidon/halo2_poseidon.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/zk/base/transcripts/transcript.h"

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
      : TranscriptReader<AffinePointTy>(std::move(buffer)),
        state_(crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63,
                                                                 0)) {}

  // Transcript methods
  Challenge255<ScalarField> SqueezeChallenge() override {
    return Challenge255<ScalarField>(state_.SqueezeNativeFieldElements(1)[0]);
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    std::array<ScalarField, 2> coords = {CurveConfig::BaseToScalar(point.x()),
                                         CurveConfig::BaseToScalar(point.y())};
    return state_.Absorb(coords);
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    return state_.Absorb(scalar);
  }

 private:
  crypto::Halo2PoseidonSponge<ScalarField> state_;
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
      : TranscriptWriter<AffinePointTy>(std::move(buffer)),
        state_(crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63,
                                                                 0)) {}

  // Transcript methods
  Challenge255<ScalarField> SqueezeChallenge() override {
    return Challenge255<ScalarField>(state_.SqueezeNativeFieldElements(1)[0]);
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    std::array<ScalarField, 2> coords = {CurveConfig::BaseToScalar(point.x()),
                                         CurveConfig::BaseToScalar(point.y())};
    return state_.Absorb(coords);
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    return state_.Absorb(scalar);
  }

 private:
  crypto::Halo2PoseidonSponge<ScalarField> state_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_TRANSCRIPTS_POSEIDON_TRANSCRIPT_H_
