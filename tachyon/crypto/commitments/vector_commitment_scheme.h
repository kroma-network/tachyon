#ifndef TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/base/bits.h"
#include "tachyon/crypto/commitments/batch_commitment_state.h"
#include "tachyon/crypto/commitments/vector_commitment_scheme_traits_forward.h"

namespace tachyon::crypto {

template <typename Derived>
class VectorCommitmentScheme {
 public:
  constexpr static size_t kMaxSize =
      VectorCommitmentSchemeTraits<Derived>::kMaxSize;
  constexpr static bool kIsTransparent =
      VectorCommitmentSchemeTraits<Derived>::kIsTransparent;
  constexpr static bool kSupportsBatchMode =
      VectorCommitmentSchemeTraits<Derived>::kSupportsBatchMode;

  using Field = typename VectorCommitmentSchemeTraits<Derived>::Field;
  using Commitment = typename VectorCommitmentSchemeTraits<Derived>::Commitment;

  uint32_t K() const {
    const Derived* derived = static_cast<const Derived*>(this);
    return base::bits::SafeLog2Ceiling(derived->N());
  }

  BatchCommitmentState& batch_commitment_state() {
    return batch_commitment_state_;
  }
  bool GetBatchMode() const { return batch_commitment_state_.batch_mode; }

  template <typename T = Derived, std::enable_if_t<VectorCommitmentSchemeTraits<
                                      T>::kSupportsBatchMode>* = nullptr>
  void SetBatchMode(size_t batch_count) {
    if (batch_count == 0) return;
    CHECK_EQ(batch_commitment_state_.batch_count, size_t{0});
    batch_commitment_state_ = BatchCommitmentState(true, batch_count);
    Derived* derived = static_cast<Derived*>(this);
    derived->ResizeBatchCommitments();
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

  template <typename T = Derived, typename Params,
            std::enable_if_t<
                !VectorCommitmentSchemeTraits<T>::kIsTransparent>* = nullptr>
  [[nodiscard]] bool UnsafeSetup(size_t size, const Params& params) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoUnsafeSetup(size, params);
  }

  // Commit to |container| and populates |result| with the commitment.
  // Return false if the size of |container| doesn't match with the size of
  // parameters.
  template <typename Container>
  [[nodiscard]] bool Commit(const Container& container,
                            Commitment* result) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(container, result);
  }

  // Commit to |container| and stores the commitment in |batch_commitments_| at
  // |index| if |batch_mode| is true. Return false if the size of |container|
  // doesn't match with the size of parameters. It terminates when |batch_mode|
  // is false.
  template <typename T = Derived, typename Container,
            std::enable_if_t<
                VectorCommitmentSchemeTraits<T>::kSupportsBatchMode>* = nullptr>
  [[nodiscard]] bool Commit(const Container& container, size_t index) {
    Derived* derived = static_cast<Derived*>(this);
    CHECK(batch_commitment_state_.batch_mode);
    return derived->DoCommit(container, batch_commitment_state_, index);
  }

  // Commit to |container| with a |random_value| and populates |result| with the
  // commitment. Return false if the size of |container| doesn't match with the
  // size of parameters.
  template <typename Container>
  [[nodiscard]] bool Commit(const Container& container,
                            const Field& random_value,
                            Commitment* result) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(container, random_value, result);
  }

  // Commit to |container| with a |random_value| and stores the commitment in
  // |batch_commitments_| at |index| if |batch_mode| is true. Return false if
  // the size of |container| doesn't match with the size of parameters. It
  // terminates when |batch_mode| is false.
  template <typename T = Derived, typename Container,
            std::enable_if_t<
                VectorCommitmentSchemeTraits<T>::kSupportsBatchMode>* = nullptr>
  [[nodiscard]] bool Commit(const Container& container,
                            const Field& random_value, size_t index) {
    Derived* derived = static_cast<Derived*>(this);
    CHECK(batch_commitment_state_.batch_mode);
    return derived->DoCommit(container, random_value, batch_commitment_state_,
                             index);
  }

  // Create an opening proof that proves that |members| belong to a
  // commitment.
  template <typename Container, typename Proof>
  [[nodiscard]] bool CreateOpeningProof(const Container& members,
                                        Proof* proof) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCreateOpeningProof(members, proof);
  }

  // Verify an opening |proof| that proves that |members| belong to a
  // |commitment|.
  // NOTE(chokobole): const was removed from |Commitment| since it can be a
  // |Transcript|. At this moment, |WriteToTranscript()| is not a const method.
  template <typename Container, typename Proof>
  [[nodiscard]] bool VerifyOpeningProof(Commitment& commitment,
                                        const Container& members,
                                        const Proof& proof) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoVerifyOpeningProof(commitment, members, proof);
  }

  // Verify multi-openings |proof|.
  template <typename Container, typename Proof>
  [[nodiscard]] bool VerifyOpeningProof(const Container& members,
                                        Proof* proof) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoVerifyOpeningProof(members, proof);
  }

 protected:
  BatchCommitmentState batch_commitment_state_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_
