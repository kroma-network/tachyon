#ifndef TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_

#include "tachyon/base/bits.h"
#include "tachyon/crypto/commitments/vector_commitment_scheme_traits_forward.h"

namespace tachyon::crypto {

template <typename Derived>
class VectorCommitmentScheme {
 public:
  constexpr static size_t kMaxSize =
      VectorCommitmentSchemeTraits<Derived>::kMaxSize;
  constexpr static bool kIsTransparent =
      VectorCommitmentSchemeTraits<Derived>::kIsTransparent;

  using Field = typename VectorCommitmentSchemeTraits<Derived>::Field;
  using Commitment = typename VectorCommitmentSchemeTraits<Derived>::Commitment;
  using TranscriptReader =
      typename VectorCommitmentSchemeTraits<Derived>::TranscriptReader;
  using TranscriptWriter =
      typename VectorCommitmentSchemeTraits<Derived>::TranscriptWriter;
  using Proof = typename VectorCommitmentSchemeTraits<Derived>::Proof;

  size_t K() const {
    const Derived* derived = static_cast<const Derived*>(this);
    return base::bits::SafeLog2Ceiling(derived->N());
  }

  // Initialize parameters.
  template <typename T = Derived,
            std::enable_if_t<VectorCommitmentSchemeTraits<T>::kIsTransparent>* =
                nullptr>
  [[nodiscard]] bool Setup() {
    return Setup(kMaxSize);
  }

  template <typename T = Derived,
            std::enable_if_t<VectorCommitmentSchemeTraits<T>::kIsTransparent>* =
                nullptr>
  [[nodiscard]] bool Setup(size_t size) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoSetup(size);
  }

  // Initialize parameters.
  template <typename T = Derived,
            std::enable_if_t<
                !VectorCommitmentSchemeTraits<T>::kIsTransparent>* = nullptr>
  [[nodiscard]] bool UnsafeSetup() {
    return UnsafeSetup(kMaxSize);
  }

  template <typename T = Derived,
            std::enable_if_t<
                !VectorCommitmentSchemeTraits<T>::kIsTransparent>* = nullptr>
  [[nodiscard]] bool UnsafeSetup(size_t size) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoUnsafeSetup(size);
  }

  template <typename Params, typename T = Derived,
            std::enable_if_t<
                !VectorCommitmentSchemeTraits<T>::kIsTransparent>* = nullptr>
  [[nodiscard]] bool UnsafeSetup(size_t size, const Params& params) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoUnsafeSetup(size, params);
  }

  // Commit to |container| and populates |commitment|.
  // Return false if the size of |container| doesn't match with the size of
  // parameters.
  template <typename Container, typename T = Derived,
            std::enable_if_t<!VectorCommitmentSchemeTraits<
                T>::kIsCommitInteractive>* = nullptr>
  [[nodiscard]] bool Commit(const Container& container,
                            Commitment* commitment) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(container, commitment);
  }

  // Commit to |container| with a |random_value| and populates |commitment|.
  // Return false if the size of |container| doesn't match with the size of
  // parameters. e.g, Pedersen.
  template <typename Container, typename T = Derived,
            std::enable_if_t<!VectorCommitmentSchemeTraits<
                T>::kIsCommitInteractive>* = nullptr>
  [[nodiscard]] bool Commit(const Container& container,
                            const Field& random_value,
                            Commitment* commitment) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(container, random_value, commitment);
  }

  // Commit to |container| with a |transcript_writer| and populates
  // |commitment|. Return false if the size of |container| doesn't match with
  // the size of parameters.
  template <
      typename Container, typename T = Derived,
      std::enable_if_t<VectorCommitmentSchemeTraits<T>::IsCommitInteractive>* =
          nullptr>
  [[nodiscard]] bool Commit(const Container& container, Commitment* commitment,
                            TranscriptWriter* transcript_writer) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(container, commitment, transcript_writer);
  }

  // Create an opening |proof| based on |args|. |args| is defined according to
  // the implementation.
  template <
      typename Args, typename T = Derived,
      std::enable_if_t<!VectorCommitmentSchemeTraits<T>::kIsOpenInteractive>* =
          nullptr>
  [[nodiscard]] bool CreateOpeningProof(const Args& args, Proof* proof) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCreateOpeningProof(args, proof);
  }

  // Create an opening |proof| with a |transcript_writer| based on |args|.
  // |args| is defined according to the implementation.
  template <typename Args, typename T = Derived,
            std::enable_if_t<
                VectorCommitmentSchemeTraits<T>::kIsOpenInteractive>* = nullptr>
  [[nodiscard]] bool CreateOpeningProof(
      const Args& args, Proof* proof,
      TranscriptWriter* transcript_writer) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCreateOpeningProof(args, proof, transcript_writer);
  }

  // Verify an opening |proof| with |args|. e.g, BinaryMerkleTree.
  // |args| is defined according to the implementation.
  template <
      typename Args, typename T = Derived,
      std::enable_if_t<
          !(VectorCommitmentSchemeTraits<T>::kIsCommitInteractive ||
            VectorCommitmentSchemeTraits<T>::kIsOpenInteractive)>* = nullptr>
  [[nodiscard]] bool VerifyOpeningProof(const Args& args,
                                        const Proof& proof) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoVerifyOpeningProof(args, proof);
  }

  // Verify an opening |proof| with a |transcript_reader| and |args|. e.g, FRI.
  // |args| is defined according to the implementation.
  template <typename Args, typename T = Derived,
            std::enable_if_t<
                VectorCommitmentSchemeTraits<T>::kIsCommitInteractive ||
                VectorCommitmentSchemeTraits<T>::kIsOpenInteractive>* = nullptr>
  [[nodiscard]] bool VerifyOpeningProof(
      const Args& args, const Proof& proof,
      TranscriptReader& transcript_reader) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoVerifyOpeningProof(args, proof, transcript_reader);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_
