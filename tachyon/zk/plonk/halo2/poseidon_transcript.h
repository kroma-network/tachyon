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
#include "tachyon/zk/plonk/halo2/proof_serializer.h"

namespace tachyon {
namespace zk::halo2 {

template <typename Curve>
class PoseidonReader;

template <typename Curve>
class PoseidonWriter;

namespace internal {

template <typename Curve>
class PoseidonBase {
 protected:
  friend class crypto::Transcript<PoseidonReader<Curve>>;
  friend class crypto::Transcript<PoseidonWriter<Curve>>;

  using Point = math::AffinePoint<Curve>;
  using BaseField = typename Point::BaseField;
  using ScalarField = typename Point::ScalarField;

  PoseidonBase()
      : state_(crypto::PoseidonConfig<ScalarField>::CreateCustom(8, 5, 8, 63,
                                                                 0)) {}

  ScalarField DoSqueezeChallenge() {
    return state_.SqueezeNativeFieldElements(1)[0];
  }

  bool DoWriteToTranscript(const Point& point) {
    std::array<ScalarField, 2> coords = {
        Curve::Config::BaseToScalar(point.x()),
        Curve::Config::BaseToScalar(point.y())};
    return state_.Absorb(coords);
  }

  bool DoWriteToTranscript(const ScalarField& scalar) {
    return state_.Absorb(scalar);
  }

  PoseidonSponge<ScalarField> state_;
};

}  // namespace internal

template <typename Curve>
class PoseidonReader final
    : public crypto::TranscriptReader<PoseidonReader<Curve>>,
      public internal::PoseidonBase<Curve> {
 public:
  // Initialize a transcript given an input buffer.
  explicit PoseidonReader(base::Buffer buffer)
      : crypto::TranscriptReader<PoseidonReader<Curve>>(std::move(buffer)) {}

 private:
  friend class crypto::TranscriptReader<PoseidonReader<Curve>>;

  template <typename T>
  bool DoReadFromProof(T* value) const {
    return ProofSerializer<T>::ReadFromProof(this->buffer_, value);
  }
};

template <typename Curve>
class PoseidonWriter final
    : public crypto::TranscriptWriter<PoseidonWriter<Curve>>,
      public internal::PoseidonBase<Curve> {
 public:
  // Initialize a transcript given an output buffer.
  explicit PoseidonWriter(base::Uint8VectorBuffer buffer)
      : crypto::TranscriptWriter<PoseidonWriter<Curve>>(std::move(buffer)) {}

 private:
  friend class crypto::TranscriptWriter<PoseidonWriter<Curve>>;

  template <typename T>
  bool DoWriteToProof(const T& value) {
    return ProofSerializer<T>::WriteToProof(value, this->buffer_);
  }
};

}  // namespace zk::halo2

namespace crypto {

template <typename Curve>
struct TranscriptTraits<zk::halo2::PoseidonReader<Curve>> {
  using Field = typename math::AffinePoint<Curve>::ScalarField;
};

template <typename Curve>
struct TranscriptTraits<zk::halo2::PoseidonWriter<Curve>> {
  using Field = typename math::AffinePoint<Curve>::ScalarField;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_POSEIDON_TRANSCRIPT_H_
