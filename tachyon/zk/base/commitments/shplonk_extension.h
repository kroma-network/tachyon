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

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename Commitment>
class SHPlonkExtension final
    : public UnivariatePolynomialCommitmentSchemeExtension<
          SHPlonkExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>> {
 public:
  // NOTE(dongchangYoo): The following value are pre-determined according to
  // the Commitment Opening Scheme.
  // https://
  // github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/poly/kzg/multiopen/shplonk/prover.rs#L111
  constexpr static bool kQueryInstance = false;

  using Base = UnivariatePolynomialCommitmentSchemeExtension<
      SHPlonkExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>>;
  using Field = typename Base::Field;
  using Poly = typename Base::Poly;
  using Evals = typename Base::Evals;

  SHPlonkExtension() = default;
  explicit SHPlonkExtension(
      crypto::SHPlonk<Curve, MaxDegree, Commitment>&& shplonk)
      : shplonk_(std::move(shplonk)) {}

  size_t N() const { return shplonk_.N(); }

  size_t D() const { return N() - 1; }

  [[nodiscard]] bool DoUnsafeSetup(size_t size) {
    return shplonk_.DoUnsafeSetup(size);
  }

  [[nodiscard]] bool DoUnsafeSetup(size_t size, const Field& tau) {
    return shplonk_.DoUnsafeSetup(size, tau);
  }

  template <typename BaseContainerTy>
  [[nodiscard]] bool DoCommit(const BaseContainerTy& v, Commitment* out) const {
    return shplonk_.DoCommit(v, out);
  }

  [[nodiscard]] bool DoCommit(const Poly& poly, Commitment* out) const {
    return shplonk_.DoCommit(poly, out);
  }

  [[nodiscard]] bool DoCommitLagrange(const Evals& evals,
                                      Commitment* out) const {
    return shplonk_.DoCommitLagrange(evals, out);
  }

  template <typename BaseContainerTy>
  [[nodiscard]] bool DoCommitLagrange(const BaseContainerTy& v,
                                      Commitment* out) const {
    return shplonk_.DoCommitLagrange(v, out);
  }

  template <typename ContainerTy, typename Proof>
  [[nodiscard]] bool DoCreateOpeningProof(const ContainerTy& poly_openings,
                                          Proof* proof) const {
    return shplonk_.DoCreateOpeningProof(poly_openings, proof);
  }

 private:
  crypto::SHPlonk<Curve, MaxDegree, Commitment> shplonk_;
};

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename Commitment>
struct UnivariatePolynomialCommitmentSchemeExtensionTraits<
    SHPlonkExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>> {
 public:
  constexpr static size_t kMaxExtendedDegree = MaxExtendedDegree;
  constexpr static size_t kMaxExtendedSize = kMaxExtendedDegree + 1;
};

}  // namespace zk

namespace crypto {

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename _Commitment>
struct VectorCommitmentSchemeTraits<
    zk::SHPlonkExtension<Curve, MaxDegree, MaxExtendedDegree, _Commitment>> {
 public:
  using G1PointTy = typename Curve::G1Curve::AffinePointTy;
  using Field = typename G1PointTy::ScalarField;
  using Commitment = _Commitment;

  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = false;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_ZK_BASE_COMMITMENTS_SHPLONK_EXTENSION_H_
