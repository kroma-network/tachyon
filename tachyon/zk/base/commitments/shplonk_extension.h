// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_COMMITMENTS_SHPLONK_EXTENSION_H_
#define TACHYON_ZK_BASE_COMMITMENTS_SHPLONK_EXTENSION_H_

#include <utility>

#include "tachyon/crypto/commitments/kzg/shplonk.h"
#include "tachyon/zk/base/commitments/univariate_polynomial_commitment_scheme_extension.h"

namespace tachyon {
namespace zk {

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          size_t MaxExtendedDegree, typename Commitment>
class SHPlonkExtension
    : public UnivariatePolynomialCommitmentSchemeExtension<SHPlonkExtension<
          G1PointTy, G2PointTy, MaxDegree, MaxExtendedDegree, Commitment>> {
 public:
  using Field = typename G1PointTy::ScalarField;

  SHPlonkExtension() = default;
  explicit SHPlonkExtension(
      crypto::SHPlonk<G1PointTy, G2PointTy, MaxDegree, Commitment>&& shplonk)
      : shplonk_(std::move(shplonk)) {}

  size_t N() const { return shplonk_.N(); }

  [[nodiscard]] bool DoUnsafeSetup(size_t size) {
    return shplonk_.DoUnsafeSetup(size);
  }

  template <typename BaseContainerTy>
  [[nodiscard]] bool DoCommit(const BaseContainerTy& v, Commitment* out) const {
    return shplonk_.DoCommit(v, out);
  }

  template <typename BaseContainerTy>
  [[nodiscard]] bool DoCommitLagrange(const BaseContainerTy& v,
                                      Commitment* out) const {
    return shplonk_.DoCommitLagrange(v, out);
  }

  template <typename ContainerTy>
  [[nodiscard]] bool DoCreateMultiOpeningProof(
      const ContainerTy& poly_openings,
      crypto::SHPlonkProof<Commitment>* proof) const {
    return shplonk_.DoCreateMultiOpeningProof(poly_openings, proof);
  }

 private:
  crypto::SHPlonk<G1PointTy, G2PointTy, MaxDegree, Commitment> shplonk_;
};

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          size_t MaxExtendedDegree, typename Commitment>
struct UnivariatePolynomialCommitmentSchemeExtensionTraits<SHPlonkExtension<
    G1PointTy, G2PointTy, MaxDegree, MaxExtendedDegree, Commitment>> {
 public:
  constexpr static size_t kMaxExtendedDegree = MaxExtendedDegree;
  constexpr static size_t kMaxExtendedSize = kMaxExtendedDegree + 1;
};

}  // namespace zk

namespace crypto {

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          size_t MaxExtendedDegree, typename _Commitment>
struct VectorCommitmentSchemeTraits<zk::SHPlonkExtension<
    G1PointTy, G2PointTy, MaxDegree, MaxExtendedDegree, _Commitment>> {
 public:
  using Field = typename G1PointTy::ScalarField;
  using Commitment = _Commitment;

  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = false;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_ZK_BASE_COMMITMENTS_SHPLONK_EXTENSION_H_
