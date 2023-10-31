// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_TRANSCRIPT_TRANSCRIPT_H_
#define TACHYON_ZK_TRANSCRIPT_TRANSCRIPT_H_

#include "tachyon/math/base/big_int.h"

namespace tachyon::zk {

// A 255-bit challenge.
template <typename AffinePointTy>
class Challenge255 {
 public:
  using ScalarField = typename AffinePointTy::ScalarField;
  static_assert(ScalarField::N <= 4);

  constexpr Challenge255() = default;
  constexpr explicit Challenge255(const ScalarField& challenge_input)
      : challenge_(math::BigInt<4>::FromBytesLE(
            challenge_input.ToBigInt().ToBytesLE())) {}

  constexpr const math::BigInt<4>& challenge() const { return challenge_; }

  constexpr ScalarField ChallengeAsScalar() const {
    return ScalarField::FromBigInt(challenge_);
  }

 private:
  math::BigInt<4> challenge_;
};

// Generic transcript view (from either the prover or verifier's perspective)
template <typename AffinePointTy>
class Transcript {
 public:
  using ScalarField = typename AffinePointTy::ScalarField;

  virtual ~Transcript() = default;

  // Squeeze an encoded verifier challenge from the transcript.
  virtual Challenge255<AffinePointTy> SqueezeChallenge() = 0;

  // Write a curve |point| to the transcript without writing it to the proof,
  // treating it as a common input.
  virtual bool WriteToTranscript(const AffinePointTy& point) = 0;

  // Write a curve |scalar| to the transcript without writing it to the proof,
  // treating it as a common input.
  virtual bool WriteToTranscript(const ScalarField& scalar) = 0;
};

// Transcript view from the perspective of a verifier that has access to an
// input stream of data from the prover to the verifier.
template <typename AffinePointTy>
class TranscriptRead : public Transcript<AffinePointTy> {
 public:
  using ScalarField = typename AffinePointTy::ScalarField;

  // Read a curve |point| from the prover. Note that it also writes the
  // |point| to the transcript by calling |WriteToTranscript()| internally.
  virtual bool ReadPoint(AffinePointTy* point) = 0;

  // Read a curve |scalar| from the prover. Note that it also writes the
  // |scalar| to the transcript by calling |WriteToTranscript()| internally.
  virtual bool ReadScalar(ScalarField* scalar) = 0;
};

// Transcript view from the perspective of a prover that has access to an output
// stream of messages from the prover to the verifier.
template <typename AffinePointTy>
class TranscriptWrite : public Transcript<AffinePointTy> {
 public:
  using ScalarField = typename AffinePointTy::ScalarField;

  // Write a curve |point| to the proof. Note that it also writes the
  // |point| to the transcript by calling |WriteToTranscript()| internally.
  virtual bool WriteToProof(const AffinePointTy& point) = 0;

  // Write a curve |scalar| to the proof. Note that it also writes the
  // |scalar| to the transcript by calling |WriteToTranscript()| internally.
  virtual bool WriteToProof(const ScalarField& scalar) = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_TRANSCRIPT_TRANSCRIPT_H_
