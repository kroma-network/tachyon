// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_TRANSCRIPTS_BLAKE2B_TRANSCRIPT_H_
#define TACHYON_CRYPTO_TRANSCRIPTS_BLAKE2B_TRANSCRIPT_H_

#include <array>
#include <utility>

#include "openssl/blake2.h"

#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/elliptic_curves/affine_point.h"

namespace tachyon::crypto {

// Prefix to a prover's message soliciting a challenge
constexpr uint8_t kBlake2bPrefixChallenge[1] = {0};

// Prefix to a prover's message containing a curve point
constexpr uint8_t kBlake2bPrefixPoint[1] = {1};

// Prefix to a prover's message containing a scalar
constexpr uint8_t kBlake2bPrefixScalar[1] = {2};

// TODO(TomTaehoonKim): We will replace Blake2b with an algebraic hash function
// in a later version. See
// https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/transcript/blake2b.rs#L25
template <typename Curve>
class Blake2bReader : public TranscriptReader<math::AffinePoint<Curve>> {
 public:
  using AffinePointTy = typename Curve::AffinePointTy;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;

  Blake2bReader() = default;
  // Initialize a transcript given an input buffer.
  explicit Blake2bReader(base::Buffer read_buf)
      : TranscriptReader<AffinePointTy>(std::move(read_buf)) {
    BLAKE2B512_Init(&state_);
  }
  // Initialize a transcript given an input buffer and |personal|.
  explicit Blake2bReader(base::Buffer read_buf, std::string_view personal)
      : TranscriptReader<AffinePointTy>(std::move(read_buf)) {
    BLAKE2B512_InitWithPersonal(&state_, personal.data());
  }

  // Transcript methods
  Challenge255<ScalarField> SqueezeChallenge() override {
    BLAKE2B512_Update(&state_, kBlake2bPrefixChallenge, 1);
    BLAKE2B_CTX hasher = state_;
    uint8_t result[64] = {0};
    BLAKE2B512_Final(result, &hasher);
    return Challenge255<ScalarField>(
        ScalarField::FromAnySizedBigInt(math::BigInt<8>::FromBytesLE(result)));
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    BLAKE2B512_Update(&state_, kBlake2bPrefixPoint, 1);
    if (point.infinity()) {
      BLAKE2B512_Update(&state_, BaseField::BigIntTy::Zero().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
      BLAKE2B512_Update(&state_,
                        typename BaseField::BigIntTy(5).ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
    } else {
      BLAKE2B512_Update(&state_, point.x().ToBigInt().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
      BLAKE2B512_Update(&state_, point.y().ToBigInt().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
    }
    return true;
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    BLAKE2B512_Update(&state_, kBlake2bPrefixScalar, 1);
    BLAKE2B512_Update(&state_, scalar.ToBigInt().ToBytesLE().data(),
                      ScalarField::BigIntTy::kByteNums);
    return true;
  }

 private:
  BLAKE2B_CTX state_;
};

template <typename Curve>
class Blake2bWriter : public TranscriptWriter<math::AffinePoint<Curve>> {
 public:
  using AffinePointTy = typename Curve::AffinePointTy;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;

  Blake2bWriter() = default;
  // Initialize a transcript given an output buffer.
  explicit Blake2bWriter(base::VectorBuffer write_buf)
      : TranscriptWriter<AffinePointTy>(std::move(write_buf)) {
    BLAKE2B512_Init(&state_);
  }
  // Initialize a transcript given an output buffer and |personal|.
  Blake2bWriter(base::VectorBuffer write_buf, std::string_view personal)
      : TranscriptWriter<AffinePointTy>(std::move(write_buf)) {
    BLAKE2B512_InitWithPersonal(&state_, personal.data());
  }

  // Transcript methods
  Challenge255<ScalarField> SqueezeChallenge() override {
    BLAKE2B512_Update(&state_, kBlake2bPrefixChallenge, 1);
    BLAKE2B_CTX hasher = state_;
    uint8_t result[64] = {0};
    BLAKE2B512_Final(result, &hasher);
    return Challenge255<ScalarField>(
        ScalarField::FromAnySizedBigInt(math::BigInt<8>::FromBytesLE(result)));
  }

  bool WriteToTranscript(const AffinePointTy& point) override {
    BLAKE2B512_Update(&state_, kBlake2bPrefixPoint, 1);
    if (point.infinity()) {
      BLAKE2B512_Update(&state_, BaseField::BigIntTy::Zero().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
      BLAKE2B512_Update(&state_,
                        typename BaseField::BigIntTy(5).ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
    } else {
      BLAKE2B512_Update(&state_, point.x().ToBigInt().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
      BLAKE2B512_Update(&state_, point.y().ToBigInt().ToBytesLE().data(),
                        BaseField::BigIntTy::kByteNums);
    }
    return true;
  }

  bool WriteToTranscript(const ScalarField& scalar) override {
    BLAKE2B512_Update(&state_, kBlake2bPrefixScalar, 1);
    BLAKE2B512_Update(&state_, scalar.ToBigInt().ToBytesLE().data(),
                      ScalarField::BigIntTy::kByteNums);
    return true;
  }

 private:
  BLAKE2B_CTX state_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_TRANSCRIPTS_BLAKE2B_TRANSCRIPT_H_
