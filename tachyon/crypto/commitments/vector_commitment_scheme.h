#ifndef TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_

#include "tachyon/base/bits.h"
#include "tachyon/crypto/commitments/vector_commitment_scheme_traits.h"

namespace tachyon::crypto {

template <typename C>
class VectorCommitmentScheme {
 public:
  constexpr static size_t kMaxSize = VectorCommitmentSchemeTraits<C>::kMaxSize;
  constexpr static bool kIsTransparent =
      VectorCommitmentSchemeTraits<C>::kIsTransparent;

  using Field = typename VectorCommitmentSchemeTraits<C>::Field;
  using ResultTy = typename VectorCommitmentSchemeTraits<C>::ResultTy;

  size_t K() const {
    const C* c = static_cast<const C*>(this);
    return base::bits::SafeLog2Ceiling(c->N());
  }

  // Initialize parameters.
  template <typename C2 = C, std::enable_if_t<VectorCommitmentSchemeTraits<
                                 C2>::kIsTransparent>* = nullptr>
  [[nodiscard]] bool Setup() {
    return Setup(kMaxSize);
  }

  template <typename C2 = C, std::enable_if_t<VectorCommitmentSchemeTraits<
                                 C2>::kIsTransparent>* = nullptr>
  [[nodiscard]] bool Setup(size_t size) {
    C* c = static_cast<C*>(this);
    return c->DoSetup(size);
  }

  // Initialize parameters.
  template <typename C2 = C, std::enable_if_t<!VectorCommitmentSchemeTraits<
                                 C2>::kIsTransparent>* = nullptr>
  [[nodiscard]] bool UnsafeSetup() {
    return UnsafeSetup(kMaxSize);
  }

  template <typename C2 = C, std::enable_if_t<!VectorCommitmentSchemeTraits<
                                 C2>::kIsTransparent>* = nullptr>
  [[nodiscard]] bool UnsafeSetup(size_t size) {
    C* c = static_cast<C*>(this);
    return c->DoUnsafeSetup(size);
  }

  // Commit to |container| and populates |result| with the commitment.
  // Return false if the size of |container| doesn't match with the size of
  // parameters.
  template <typename ContainerTy>
  [[nodiscard]] bool Commit(const ContainerTy& container,
                            ResultTy* result) const {
    const C* c = static_cast<const C*>(this);
    return c->DoCommit(container, result);
  }

  // Commit to |container| with a |random_value| and populates |result| with the
  // commitment. Return false if the size of |container| doesn't match with the
  // size of parameters.
  template <typename ContainerTy>
  [[nodiscard]] bool Commit(const ContainerTy& container,
                            const Field& random_value, ResultTy* result) const {
    const C* c = static_cast<const C*>(this);
    return c->DoCommit(container, random_value, result);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_VECTOR_COMMITMENT_SCHEME_H_
