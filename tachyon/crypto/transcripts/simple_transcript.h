#ifndef TACHYON_CRYPTO_TRANSCRIPTS_SIMPLE_TRANSCRIPT_H_
#define TACHYON_CRYPTO_TRANSCRIPTS_SIMPLE_TRANSCRIPT_H_

#include <type_traits>
#include <utility>

#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::crypto {

template <typename F, typename SFINAE = void>
class SimpleTranscriptReader;

template <typename F, typename SFINAE = void>
class SimpleTranscriptWriter;

namespace internal {

template <typename F, typename SFINAE = void>
class SimpleTranscript;

template <typename F>
class SimpleTranscript<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>> {
 protected:
  friend class Transcript<SimpleTranscriptReader<F>>;
  friend class Transcript<SimpleTranscriptWriter<F>>;

  F DoSqueezeChallenge() { return state_.DoubleInPlace(); }

  bool DoWriteToTranscript(const F& value) {
    state_ += value;
    return true;
  }

 private:
  F state_ = F::Zero();
};

template <typename Curve>
class SimpleTranscript<math::AffinePoint<Curve>> {
 protected:
  friend class Transcript<SimpleTranscriptReader<math::AffinePoint<Curve>>>;
  friend class Transcript<SimpleTranscriptWriter<math::AffinePoint<Curve>>>;

  using F = typename math::AffinePoint<Curve>::ScalarField;
  using CurveConfig = typename Curve::Config;

  F DoSqueezeChallenge() { return state_.DoubleInPlace(); }

  bool DoWriteToTranscript(const math::AffinePoint<Curve>& point) {
    state_ += CurveConfig::BaseToScalar(point.x());
    state_ += CurveConfig::BaseToScalar(point.y());
    return true;
  }

  bool DoWriteToTranscript(const F& scalar) {
    state_ += scalar;
    return true;
  }

 private:
  F state_ = F::Zero();
};

}  // namespace internal

template <typename F>
class SimpleTranscriptReader<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>>
    final : public TranscriptReader<SimpleTranscriptReader<F>>,
            public internal::SimpleTranscript<F> {
 public:
  explicit SimpleTranscriptReader(base::Buffer read_buf)
      : TranscriptReader<SimpleTranscriptReader<F>>(std::move(read_buf)) {}

 private:
  friend class TranscriptReader<SimpleTranscriptReader<F>>;

  template <typename T>
  bool DoReadFromProof(T* value) const {
    return this->buffer_.Read(value);
  }
};

template <typename F>
struct TranscriptTraits<SimpleTranscriptReader<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>>> {
  using Field = F;
};

template <typename Curve>
class SimpleTranscriptReader<math::AffinePoint<Curve>> final
    : public TranscriptReader<SimpleTranscriptReader<math::AffinePoint<Curve>>>,
      public internal::SimpleTranscript<math::AffinePoint<Curve>> {
 public:
  explicit SimpleTranscriptReader(base::Buffer read_buf)
      : TranscriptReader<SimpleTranscriptReader<math::AffinePoint<Curve>>>(
            std::move(read_buf)) {}

 private:
  friend class TranscriptReader<
      SimpleTranscriptReader<math::AffinePoint<Curve>>>;

  template <typename T>
  bool DoReadFromProof(T* value) const {
    return this->buffer_.Read(value);
  }
};

template <typename Curve>
struct TranscriptTraits<SimpleTranscriptReader<math::AffinePoint<Curve>>> {
  using Field = typename math::AffinePoint<Curve>::ScalarField;
};

template <typename F>
class SimpleTranscriptWriter<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>>
    : public TranscriptWriter<SimpleTranscriptWriter<F>>,
      public internal::SimpleTranscript<F> {
 public:
  using Field = F;

  explicit SimpleTranscriptWriter(base::Uint8VectorBuffer write_buf)
      : TranscriptWriter<SimpleTranscriptWriter<F>>(std::move(write_buf)) {}

 private:
  friend class TranscriptWriter<SimpleTranscriptWriter<F>>;

  template <typename T>
  bool DoWriteToProof(const T& value) {
    return this->buffer_.Write(value);
  }
};

template <typename F>
struct TranscriptTraits<SimpleTranscriptWriter<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>>> {
  using Field = F;
};

template <typename Curve>
class SimpleTranscriptWriter<math::AffinePoint<Curve>>
    : public TranscriptWriter<SimpleTranscriptWriter<math::AffinePoint<Curve>>>,
      public internal::SimpleTranscript<math::AffinePoint<Curve>> {
 public:
  explicit SimpleTranscriptWriter(base::Uint8VectorBuffer write_buf)
      : TranscriptWriter<SimpleTranscriptWriter<math::AffinePoint<Curve>>>(
            std::move(write_buf)) {}

 private:
  friend class TranscriptWriter<
      SimpleTranscriptWriter<math::AffinePoint<Curve>>>;

  template <typename T>
  bool DoWriteToProof(const T& value) {
    return this->buffer_.Write(value);
  }
};

template <typename Curve>
struct TranscriptTraits<SimpleTranscriptWriter<math::AffinePoint<Curve>>> {
  using Field = typename math::AffinePoint<Curve>::ScalarField;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_TRANSCRIPTS_SIMPLE_TRANSCRIPT_H_
