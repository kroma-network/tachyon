// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_H_
#define TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_H_

#include <utility>

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/transcripts/transcript_traits_forward.h"

namespace tachyon::crypto {

template <typename Derived>
class TranscriptReader;

template <typename Derived>
class TranscriptWriter;

// Generic transcript view (from either the prover or verifier's perspective)
template <typename Derived>
class Transcript {
 public:
  using Field = typename TranscriptTraits<Derived>::Field;

  // Squeeze an encoded verifier challenge from the transcript.
  Field SqueezeChallenge() {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoSqueezeChallenge();
  }

  // Write a |value| to the transcript.
  template <typename T>
  [[nodiscard]] bool WriteToTranscript(const T& value) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoWriteToTranscript(value);
  }

  TranscriptWriter<Derived>* ToWriter() {
    return static_cast<TranscriptWriter<Derived>*>(this);
  }

  TranscriptReader<Derived>* ToReader() {
    return static_cast<TranscriptReader<Derived>*>(this);
  }
};

// Transcript view from the perspective of a verifier that has access to an
// input stream of data from the prover to the verifier.
template <typename Derived>
class TranscriptReader : public Transcript<Derived> {
 public:
  TranscriptReader() = default;
  // Initialize a transcript given an input buffer.
  explicit TranscriptReader(base::Buffer read_buf)
      : buffer_(std::move(read_buf)) {}

  base::Buffer& buffer() { return buffer_; }
  const base::Buffer& buffer() const { return buffer_; }

  // Read a |value| from the proof. Note that it also writes the
  // |value| to the transcript by calling |WriteToTranscript()| internally when
  // |NeedToWriteToTranscript| is set to true.
  template <bool NeedToWriteToTranscript, typename T>
  [[nodiscard]] bool ReadFromProof(T* value) {
    Derived* derived = static_cast<Derived*>(this);
    if constexpr (NeedToWriteToTranscript) {
      return derived->DoReadFromProof(value) && this->WriteToTranscript(*value);
    } else {
      return derived->DoReadFromProof(value);
    }
  }

 protected:
  base::Buffer buffer_;
};

// Transcript view from the perspective of a prover that has access to an output
// stream of messages from the prover to the verifier.
template <typename Derived>
class TranscriptWriter : public Transcript<Derived> {
 public:
  TranscriptWriter() = default;
  // Initialize a transcript given an output buffer.
  explicit TranscriptWriter(base::Uint8VectorBuffer buf)
      : buffer_(std::move(buf)) {}

  base::Uint8VectorBuffer& buffer() { return buffer_; }
  const base::Uint8VectorBuffer& buffer() const { return buffer_; }

  base::Uint8VectorBuffer&& TakeBuffer() && { return std::move(buffer_); }

  // Write a |value| to the proof. Note that it also writes the
  // |value| to the transcript by calling |WriteToTranscript()| internally
  // when |NeedToWriteToTranscript| is set to true.
  template <bool NeedToWriteToTranscript, typename T>
  [[nodiscard]] bool WriteToProof(const T& value) {
    Derived* derived = static_cast<Derived*>(this);
    if constexpr (NeedToWriteToTranscript) {
      return this->WriteToTranscript(value) && derived->DoWriteToProof(value);
    } else {
      return derived->DoWriteToProof(value);
    }
  }

 protected:
  base::Uint8VectorBuffer buffer_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_H_
