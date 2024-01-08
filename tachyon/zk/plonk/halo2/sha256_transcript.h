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

namespace tachyon {
namespace zk::halo2 {

template <typename Curve>
class Sha256Reader;

template <typename Curve>
class Sha256Writer;

namespace internal {

template <typename Curve>
class Sha256Base {
 protected:
  friend class crypto::Transcript<Sha256Reader<Curve>>;
  friend class crypto::Transcript<Sha256Writer<Curve>>;

  using Point = math::AffinePoint<Curve>;
  using BaseField = typename Point::BaseField;
  using ScalarField = typename Point::ScalarField;

  Sha256Base() { SHA256_Init(&state_); }

  ScalarField DoSqueezeChallenge() {
    SHA256_Update(&state_, kShaPrefixChallenge, 1);
    SHA256_CTX hasher = state_;
    uint8_t result[32] = {0};
    SHA256_Final(result, &hasher);

    SHA256_Init(&state_);
    SHA256_Update(&state_, result, 32);

    if constexpr (ScalarField::N <= 4) {
      return ScalarField::FromAnySizedBigInt(
          math::BigInt<4>::FromBytesLE(result));
    } else {
      base::AlwaysFalse<Curve>();
    }
  }

  bool DoWriteToTranscript(const Point& point) {
    SHA256_Update(&state_, kShaPrefixZeros, 31);
    SHA256_Update(&state_, kShaPrefixPoint, 1);
    SHA256_Update(&state_, point.x().ToBigInt().ToBytesBE().data(),
                  BaseField::BigIntTy::kByteNums);
    SHA256_Update(&state_, point.y().ToBigInt().ToBytesBE().data(),
                  BaseField::BigIntTy::kByteNums);
    return true;
  }

  bool DoWriteToTranscript(const ScalarField& scalar) {
    SHA256_Update(&state_, kShaPrefixZeros, 31);
    SHA256_Update(&state_, kShaPrefixScalar, 1);
    SHA256_Update(&state_, scalar.ToBigInt().ToBytesBE().data(),
                  ScalarField::BigIntTy::kByteNums);
    return true;
  }

  SHA256_CTX state_;
};

}  // namespace internal

template <typename Curve>
class Sha256Reader final : public crypto::TranscriptReader<Sha256Reader<Curve>>,
                           public internal::Sha256Base<Curve> {
 public:
  // Initialize a transcript given an input buffer.
  explicit Sha256Reader(base::Buffer read_buf)
      : crypto::TranscriptReader<Sha256Reader<Curve>>(std::move(read_buf)) {}

 private:
  friend class crypto::TranscriptReader<Sha256Reader<Curve>>;

  template <typename T>
  bool DoReadFromProof(T* value) const {
    return ProofSerializer<T>::ReadFromProof(this->buffer_, value);
  }
};

template <typename Curve>
class Sha256Writer final : public crypto::TranscriptWriter<Sha256Writer<Curve>>,
                           public internal::Sha256Base<Curve> {
 public:
  // Initialize a transcript given an output buffer.
  explicit Sha256Writer(base::Uint8VectorBuffer write_buf)
      : crypto::TranscriptWriter<Sha256Writer<Curve>>(std::move(write_buf)) {
    SHA256_Init(&state_);
  }

 private:
  friend class crypto::TranscriptWriter<Sha256Writer<Curve>>;

  template <typename T>
  bool DoWriteToProof(const T& value) {
    return ProofSerializer<T>::WriteToProof(value, this->buffer_);
  }

  SHA256_CTX state_;
};

}  // namespace zk::halo2

namespace crypto {

template <typename Curve>
struct TranscriptTraits<zk::halo2::Sha256Reader<Curve>> {
  using Field = typename math::AffinePoint<Curve>::ScalarField;
};

template <typename Curve>
struct TranscriptTraits<zk::halo2::Sha256Writer<Curve>> {
  using Field = typename math::AffinePoint<Curve>::ScalarField;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_SHA256_TRANSCRIPT_H_
