// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_H_
#define TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_H_

#include <utility>

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/transcripts/transcript_traits.h"

namespace tachyon::crypto {

template <typename Commitment, bool FieldAndCommitmentAreSameType>
class TranscriptReaderImpl;

template <typename Commitment, bool FieldAndCommitmentAreSameType>
class TranscriptWriterImpl;

template <typename _Commitment, bool FieldAndCommitmentAreSameType>
class TranscriptImpl;

// Generic transcript view (from either the prover or verifier's perspective)
template <typename _Commitment>
class TranscriptImpl<_Commitment, false> {
 public:
  using Commitment = _Commitment;
  using Field = typename TranscriptTraits<Commitment>::Field;

  virtual ~TranscriptImpl() = default;

  // Squeeze an encoded verifier challenge from the transcript.
  virtual Field SqueezeChallenge() = 0;

  // Write a |commitment| to the transcript without writing it to the proof,
  // treating it as a common input.
  [[nodiscard]] virtual bool WriteToTranscript(
      const Commitment& commitment) = 0;

  // Write a |value| to the transcript without writing it to the proof,
  // treating it as a common input.
  [[nodiscard]] virtual bool WriteToTranscript(const Field& value) = 0;

  TranscriptWriterImpl<Commitment, false>* ToWriter() {
    return static_cast<TranscriptWriterImpl<Commitment, false>*>(this);
  }

  const TranscriptWriterImpl<Commitment, false>* ToWriter() const {
    return static_cast<const TranscriptWriterImpl<Commitment, false>*>(this);
  }

  TranscriptReaderImpl<Commitment, false>* ToReader() {
    return static_cast<TranscriptReaderImpl<Commitment, false>*>(this);
  }

  const TranscriptReaderImpl<Commitment, false>* ToReader() const {
    return static_cast<const TranscriptReaderImpl<Commitment, false>*>(this);
  }
};

template <typename _Field>
class TranscriptImpl<_Field, true> {
 public:
  using Field = _Field;

  virtual ~TranscriptImpl() = default;

  // Squeeze an encoded verifier challenge from the transcript.
  virtual Field SqueezeChallenge() = 0;

  // Write a |value| to the transcript without writing it to the proof,
  // treating it as a common input.
  [[nodiscard]] virtual bool WriteToTranscript(const Field& value) = 0;

  TranscriptWriterImpl<Field, true>* ToWriter() {
    return static_cast<TranscriptWriterImpl<Field, true>*>(this);
  }

  const TranscriptWriterImpl<Field, true>* ToWriter() const {
    return static_cast<const TranscriptWriterImpl<Field, true>*>(this);
  }

  TranscriptReaderImpl<Field, true>* ToReader() {
    return static_cast<TranscriptReaderImpl<Field, true>*>(this);
  }

  const TranscriptReaderImpl<Field, true>* ToReader() const {
    return static_cast<const TranscriptReaderImpl<Field, true>*>(this);
  }
};

template <typename T>
using Transcript =
    TranscriptImpl<T, TranscriptTraits<T>::kFieldAndCommitmentAreSameType>;

// Transcript view from the perspective of a verifier that has access to an
// input stream of data from the prover to the verifier.
template <typename Commitment>
class TranscriptReaderImpl<Commitment, false> : public Transcript<Commitment> {
 public:
  using Field = typename TranscriptTraits<Commitment>::Field;

  TranscriptReaderImpl() = default;
  // Initialize a transcript given an input buffer.
  explicit TranscriptReaderImpl(base::ReadOnlyBuffer read_buf)
      : buffer_(std::move(read_buf)) {}

  base::ReadOnlyBuffer& buffer() { return buffer_; }
  const base::ReadOnlyBuffer& buffer() const { return buffer_; }

  // Read a |commitment| from the proof. Note that it also writes the
  // |commitment| to the transcript by calling |WriteToTranscript()| internally.
  [[nodiscard]] bool ReadFromProof(Commitment* commitment) {
    if (!DoReadFromProof(commitment)) return false;
    VLOG(3) << "Proof[" << proof_idx_++
            << "]: " << commitment->ToHexString(true);
    return this->WriteToTranscript(*commitment);
  }

  // Read a |value| from the proof. Note that it also writes the
  // |value| to the transcript by calling |WriteToTranscript()| internally.
  [[nodiscard]] bool ReadFromProof(Field* value) {
    if (!DoReadFromProof(value)) return false;
    VLOG(3) << "Proof[" << proof_idx_++ << "]: " << value->ToHexString(true);
    return this->WriteToTranscript(*value);
  }

 protected:
  //  Read a |commitment| from the proof.
  [[nodiscard]] virtual bool DoReadFromProof(Commitment* commitment) const = 0;

  //  Read a |value| from the proof.
  [[nodiscard]] virtual bool DoReadFromProof(Field* value) const = 0;

  base::ReadOnlyBuffer buffer_;
  size_t proof_idx_ = 0;
};

template <typename Field>
class TranscriptReaderImpl<Field, true> : public Transcript<Field> {
 public:
  TranscriptReaderImpl() = default;
  // Initialize a transcript given an input buffer.
  explicit TranscriptReaderImpl(base::ReadOnlyBuffer read_buf)
      : buffer_(std::move(read_buf)) {}

  base::ReadOnlyBuffer& buffer() { return buffer_; }
  const base::ReadOnlyBuffer& buffer() const { return buffer_; }

  // Read a |value| from the proof. Note that it also writes the
  // |value| to the transcript by calling |WriteToTranscript()| internally.
  [[nodiscard]] bool ReadFromProof(Field* value) {
    if (!DoReadFromProof(value)) return false;
    VLOG(3) << "Proof[" << proof_idx_++ << "]: " << value->ToHexString(true);
    return this->WriteToTranscript(*value);
  }

 protected:
  //  Read a |value| from the proof.
  [[nodiscard]] virtual bool DoReadFromProof(Field* value) const = 0;

  base::ReadOnlyBuffer buffer_;
  size_t proof_idx_ = 0;
};

template <typename T>
using TranscriptReader =
    TranscriptReaderImpl<T,
                         TranscriptTraits<T>::kFieldAndCommitmentAreSameType>;

// Transcript view from the perspective of a prover that has access to an output
// stream of messages from the prover to the verifier.
template <typename Commitment>
class TranscriptWriterImpl<Commitment, false> : public Transcript<Commitment> {
 public:
  using Field = typename TranscriptTraits<Commitment>::Field;

  TranscriptWriterImpl() = default;
  // Initialize a transcript given an output buffer.
  explicit TranscriptWriterImpl(base::Uint8VectorBuffer buf)
      : buffer_(std::move(buf)) {}

  base::Uint8VectorBuffer& buffer() { return buffer_; }
  const base::Uint8VectorBuffer& buffer() const { return buffer_; }

  base::Uint8VectorBuffer&& TakeBuffer() && { return std::move(buffer_); }

  // Write a |commitment| to the proof. Note that it also writes the
  // |commitment| to the transcript by calling |WriteToTranscript()| internally.
  [[nodiscard]] bool WriteToProof(const Commitment& commitment) {
    VLOG(3) << "Proof[" << proof_idx_++
            << "]: " << commitment.ToHexString(true);
    return this->WriteToTranscript(commitment) && DoWriteToProof(commitment);
  }

  // Write a |value| to the proof. Note that it also writes the
  // |value| to the transcript by calling |WriteToTranscript()| internally.
  [[nodiscard]] bool WriteToProof(const Field& value) {
    VLOG(3) << "Proof[" << proof_idx_++ << "]: " << value.ToHexString(true);
    return this->WriteToTranscript(value) && DoWriteToProof(value);
  }

 protected:
  //  Write a |commitment| to the proof.
  [[nodiscard]] virtual bool DoWriteToProof(const Commitment& commitment) = 0;

  //  Write a |value| to the proof.
  [[nodiscard]] virtual bool DoWriteToProof(const Field& value) = 0;

  base::Uint8VectorBuffer buffer_;
  size_t proof_idx_ = 0;
};

// Transcript view from the perspective of a prover that has access to an output
// stream of messages from the prover to the verifier.
template <typename Field>
class TranscriptWriterImpl<Field, true> : public Transcript<Field> {
 public:
  TranscriptWriterImpl() = default;
  // Initialize a transcript given an output buffer.
  explicit TranscriptWriterImpl(base::Uint8VectorBuffer buf)
      : buffer_(std::move(buf)) {}

  base::Uint8VectorBuffer& buffer() { return buffer_; }
  const base::Uint8VectorBuffer& buffer() const { return buffer_; }

  base::Uint8VectorBuffer&& TakeBuffer() && { return std::move(buffer_); }

  // Write a |value| to the proof. Note that it also writes the
  // |value| to the transcript by calling |WriteToTranscript()| internally.
  [[nodiscard]] bool WriteToProof(const Field& value) {
    VLOG(3) << "Proof[" << proof_idx_++ << "]: " << value.ToHexString(true);
    return this->WriteToTranscript(value) && DoWriteToProof(value);
  }

 protected:
  //  Write a |value| to the proof.
  [[nodiscard]] virtual bool DoWriteToProof(const Field& value) = 0;

  base::Uint8VectorBuffer buffer_;
  size_t proof_idx_ = 0;
};

template <typename T>
using TranscriptWriter =
    TranscriptWriterImpl<T,
                         TranscriptTraits<T>::kFieldAndCommitmentAreSameType>;

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_H_
