// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_COMMITMENTS_KZG_COMMITMENT_SCHEME_EXTENSION_H_
#define TACHYON_ZK_BASE_COMMITMENTS_KZG_COMMITMENT_SCHEME_EXTENSION_H_

#include <utility>

#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/zk/base/commitments/univariate_polynomial_commitment_scheme_extension.h"

namespace tachyon {
namespace zk {

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          size_t MaxExtendedDegree, typename Commitment>
class KZGCommitmentSchemeExtension
    : public UnivariatePolynomialCommitmentSchemeExtension<
          KZGCommitmentSchemeExtension<G1PointTy, G2PointTy, MaxDegree,
                                       MaxExtendedDegree, Commitment>> {
 public:
  using Field = typename G1PointTy::ScalarField;

  KZGCommitmentSchemeExtension() = default;
  explicit KZGCommitmentSchemeExtension(
      crypto::KZGCommitmentScheme<G1PointTy, G2PointTy, MaxDegree, Commitment>&&
          kzg)
      : kzg_(std::move(kzg)) {}

  size_t N() const { return kzg_.N(); }

  [[nodiscard]] bool DoUnsafeSetup(size_t size) {
    return kzg_.DoUnsafeSetup(size);
  }

  template <typename BaseContainerTy>
  [[nodiscard]] bool DoCommit(const BaseContainerTy& v, Commitment* out) const {
    return kzg_.DoMSM(kzg_.g1_powers_of_tau_, v, out);
  }

  template <typename BaseContainerTy>
  [[nodiscard]] bool DoCommitLagrange(const BaseContainerTy& v,
                                      Commitment* out) const {
    return kzg_.DoMSM(kzg_.g1_powers_of_tau_lagrange_, v, out);
  }

 private:
  crypto::KZGCommitmentScheme<G1PointTy, G2PointTy, MaxDegree, Commitment> kzg_;
};

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          size_t MaxExtendedDegree, typename Commitment>
struct UnivariatePolynomialCommitmentSchemeExtensionTraits<
    KZGCommitmentSchemeExtension<G1PointTy, G2PointTy, MaxDegree,
                                 MaxExtendedDegree, Commitment>> {
 public:
  constexpr static size_t kMaxExtendedDegree = MaxExtendedDegree;
  constexpr static size_t kMaxExtendedSize = kMaxExtendedDegree + 1;
};

}  // namespace zk

namespace crypto {

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          size_t MaxExtendedDegree, typename _Commitment>
struct VectorCommitmentSchemeTraits<zk::KZGCommitmentSchemeExtension<
    G1PointTy, G2PointTy, MaxDegree, MaxExtendedDegree, _Commitment>> {
 public:
  using Field = typename G1PointTy::ScalarField;
  using Commitment = _Commitment;

  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = false;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_ZK_BASE_COMMITMENTS_KZG_COMMITMENT_SCHEME_EXTENSION_H_
