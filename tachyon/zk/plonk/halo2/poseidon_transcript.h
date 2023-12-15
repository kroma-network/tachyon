// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_
#define TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_

#include <array>
#include <utility>

#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/zk/plonk/halo2/poseidon_sponge.h"

namespace tachyon::zk::halo2 {

template <typename AffinePointTy>
class PoseidonReader : public crypto::TranscriptReader<AffinePointTy> {
 public:
  using ScalarField = typename AffinePointTy::ScalarField;
  using Curve = typename AffinePointTy::Curve;
  using CurveConfig = typename Curve::Config;

  PoseidonReader() = default;
  // Initialize a transcript given an input buffer.
  explicit PoseidonReader(base::Buffer buffer)
      : crypto::TranscriptReader<AffinePointTy>(std::move(buffer)),
        state_(crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63,
                                                                 0)) {}

  // crypto::TranscriptReader methods
  ScalarField SqueezeChallenge() override {
    return state_.SqueezeNativeFieldElements(1)[0];
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    std::array<ScalarField, 2> coords = {CurveConfig::BaseToScalar(point.x()),
                                         CurveConfig::BaseToScalar(point.y())};
    return state_.Absorb(coords);
  }

  bool WriteToTranscript(const ScalarField& value) override {
    return state_.Absorb(value);
  }

 private:
  PoseidonSponge<ScalarField> state_;
};

template <typename AffinePointTy>
class PoseidonWriter : public crypto::TranscriptWriter<AffinePointTy> {
 public:
  using ScalarField = typename AffinePointTy::ScalarField;
  using Curve = typename AffinePointTy::Curve;
  using CurveConfig = typename Curve::Config;

  PoseidonWriter() = default;
  // Initialize a transcript given an output buffer.
  explicit PoseidonWriter(base::Uint8VectorBuffer buffer)
      : crypto::TranscriptWriter<AffinePointTy>(std::move(buffer)),
        state_(crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63,
                                                                 0)) {}

  // crypto::TranscriptWriter methods
  ScalarField SqueezeChallenge() override {
    return state_.SqueezeNativeFieldElements(1)[0];
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    std::array<ScalarField, 2> coords = {CurveConfig::BaseToScalar(point.x()),
                                         CurveConfig::BaseToScalar(point.y())};
    return state_.Absorb(coords);
  }

  bool WriteToTranscript(const ScalarField& value) override {
    return state_.Absorb(value);
  }

 private:
  PoseidonSponge<ScalarField> state_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_
