#ifndef TACHYON_CRYPTO_TRANSCRIPTS_SIMPLE_TRANSCRIPT_H_
#define TACHYON_CRYPTO_TRANSCRIPTS_SIMPLE_TRANSCRIPT_H_

#include <utility>

#include "tachyon/crypto/transcripts/transcript.h"

namespace tachyon::crypto {
namespace internal {

template <typename F, typename SFINAE = void>
class SimpleTranscript;

template <typename F>
class SimpleTranscript<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>> {
 protected:
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
  using F = typename TranscriptTraits<math::AffinePoint<Curve>>::Field;
  using CurveConfig = typename Curve::Config;

  F DoSqueezeChallenge() { return state_.DoubleInPlace(); }

  bool DoWriteToTranscript(const math::AffinePoint<Curve>& point) {
    state_ += BaseToScalar(point.x().value());
    state_ += BaseToScalar(point.y().value());
    return true;
  }

  bool DoWriteToTranscript(const F& scalar) {
    state_ += scalar;
    return true;
  }

 private:
  template <typename BigInt>
  F BaseToScalar(const BigInt& base) {
    return F::FromMontgomery(base % F::Config::kModulus);
  }

  F state_ = F::Zero();
};

}  // namespace internal

template <typename F, typename SFINAE = void>
class SimpleTranscriptReader;

template <typename F>
class SimpleTranscriptReader<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>>
    : public TranscriptReader<F>, protected internal::SimpleTranscript<F> {
 public:
  explicit SimpleTranscriptReader(base::Buffer read_buf)
      : TranscriptReader<F>(std::move(read_buf)) {}

  // TranscriptReader methods
  F SqueezeChallenge() override { return this->DoSqueezeChallenge(); }

  bool WriteToTranscript(const F& value) override {
    return this->DoWriteToTranscript(value);
  }

 private:
  bool DoReadFromProof(F* value) const override {
    return this->buffer_.Read(value);
  }
};

template <typename Curve>
class SimpleTranscriptReader<math::AffinePoint<Curve>>
    : public TranscriptReader<math::AffinePoint<Curve>>,
      protected internal::SimpleTranscript<math::AffinePoint<Curve>> {
 public:
  using F = typename TranscriptTraits<math::AffinePoint<Curve>>::Field;

  explicit SimpleTranscriptReader(base::Buffer read_buf)
      : TranscriptReader<math::AffinePoint<Curve>>(std::move(read_buf)) {}

  // TranscriptReader methods
  F SqueezeChallenge() override { return this->DoSqueezeChallenge(); }

  bool WriteToTranscript(const math::AffinePoint<Curve>& point) override {
    return this->DoWriteToTranscript(point);
  }

  bool WriteToTranscript(const F& scalar) override {
    return this->DoWriteToTranscript(scalar);
  }

 private:
  bool DoReadFromProof(math::AffinePoint<Curve>* point) const override {
    return this->buffer_.Read(point);
  }

  bool DoReadFromProof(F* scalar) const override {
    return this->buffer_.Read(scalar);
  }
};

template <typename F, typename SFINAE = void>
class SimpleTranscriptWriter;

template <typename F>
class SimpleTranscriptWriter<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>>
    : public TranscriptWriter<F>, protected internal::SimpleTranscript<F> {
 public:
  explicit SimpleTranscriptWriter(base::Uint8VectorBuffer write_buf)
      : TranscriptWriter<F>(std::move(write_buf)) {}

  // TranscriptWriter methods
  F SqueezeChallenge() override { return this->DoSqueezeChallenge(); }

  bool WriteToTranscript(const F& value) override {
    return this->DoWriteToTranscript(value);
  }

 private:
  bool DoWriteToProof(const F& value) override {
    return this->buffer_.Write(value);
  }
};

template <typename Curve>
class SimpleTranscriptWriter<math::AffinePoint<Curve>>
    : public TranscriptWriter<math::AffinePoint<Curve>>,
      protected internal::SimpleTranscript<math::AffinePoint<Curve>> {
 public:
  using F = typename TranscriptTraits<math::AffinePoint<Curve>>::Field;

  explicit SimpleTranscriptWriter(base::Uint8VectorBuffer write_buf)
      : TranscriptWriter<math::AffinePoint<Curve>>(std::move(write_buf)) {}

  // TranscriptWriter methods
  F SqueezeChallenge() override { return this->DoSqueezeChallenge(); }

  bool WriteToTranscript(const math::AffinePoint<Curve>& point) override {
    return this->DoWriteToTranscript(point);
  }

  bool WriteToTranscript(const F& scalar) override {
    return this->DoWriteToTranscript(scalar);
  }

 private:
  bool DoWriteToProof(const math::AffinePoint<Curve>& point) override {
    return this->buffer_.Write(point);
  }

  bool DoWriteToProof(const F& scalar) override {
    return this->buffer_.Write(scalar);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_TRANSCRIPTS_SIMPLE_TRANSCRIPT_H_
