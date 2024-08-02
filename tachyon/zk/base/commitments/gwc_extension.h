// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_COMMITMENTS_GWC_EXTENSION_H_
#define TACHYON_ZK_BASE_COMMITMENTS_GWC_EXTENSION_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "tachyon/crypto/commitments/batch_commitment_state.h"
#include "tachyon/crypto/commitments/kzg/gwc.h"
#include "tachyon/zk/base/commitments/univariate_polynomial_commitment_scheme_extension.h"

namespace tachyon {
namespace c::zk::plonk::halo2 {

template <typename PCS, typename LS>
class KZGFamilyProverImpl;

}  // namespace c::zk::plonk::halo2

namespace halo2_api::bn254 {

class Prover;

}  // namespace halo2_api::bn254

namespace zk {

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename Commitment>
class GWCExtension final
    : public UnivariatePolynomialCommitmentSchemeExtension<
          GWCExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>> {
 public:
  // NOTE(dongchangYoo): The following values are pre-determined according to
  // the Commitment Opening Scheme.
  // https://github.com/kroma-network/halo2/blob/7d0a369/halo2_proofs/src/poly/kzg/multiopen/gwc/prover.rs#L35
  constexpr static bool kQueryInstance = true;

  using Base = UnivariatePolynomialCommitmentSchemeExtension<
      GWCExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>>;
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using Field = typename Base::Field;
  using Poly = typename Base::Poly;
  using Evals = typename Base::Evals;

  GWCExtension() = default;
  explicit GWCExtension(crypto::GWC<Curve, MaxDegree, Commitment>&& gwc)
      : gwc_(std::move(gwc)) {}

  GWCExtension(std::vector<G1Point, base::memory::ReusingAllocator<G1Point>>&&
                   g1_powers_of_tau,
               std::vector<G1Point, base::memory::ReusingAllocator<G1Point>>&&
                   g1_powers_of_tau_lagrange,
               G2Point&& s_g2) {
    crypto::KZG<G1Point, MaxDegree, Commitment> kzg(
        std::move(g1_powers_of_tau), std::move(g1_powers_of_tau_lagrange));
    gwc_ = crypto::GWC<Curve, MaxDegree, Commitment>(std::move(kzg),
                                                     std::move(s_g2));
  }

  const char* Name() { return gwc_.Name(); }

  size_t N() const { return gwc_.N(); }

  size_t D() const { return N() - 1; }

  const G2Point& SG2() const { return gwc_.s_g2(); }

  crypto::BatchCommitmentState& batch_commitment_state() {
    return gwc_.batch_commitment_state();
  }
  bool GetBatchMode() const { return gwc_.GetBatchMode(); }

  void SetBatchMode(size_t batch_count) { gwc_.SetBatchMode(batch_count); }

  std::vector<Commitment> GetBatchCommitments() {
    return gwc_.GetBatchCommitments();
  }

  [[nodiscard]] bool DoUnsafeSetup(size_t size) {
    return gwc_.DoUnsafeSetup(size);
  }

  [[nodiscard]] bool DoUnsafeSetup(size_t size, const Field& tau) {
    return gwc_.DoUnsafeSetup(size, tau);
  }

  template <typename ScalarContainer>
  [[nodiscard]] bool DoCommit(const ScalarContainer& v, Commitment* out) const {
    return gwc_.DoCommit(v, out);
  }

  template <typename ScalarContainer>
  [[nodiscard]] bool DoCommit(const ScalarContainer& v,
                              crypto::BatchCommitmentState& state,
                              size_t index) {
    return gwc_.DoCommit(v, state, index);
  }

  [[nodiscard]] bool DoCommit(const Poly& poly, Commitment* out) const {
    return gwc_.DoCommit(poly, out);
  }

  [[nodiscard]] bool DoCommit(const Poly& poly,
                              crypto::BatchCommitmentState& state,
                              size_t index) {
    return gwc_.DoCommit(poly, state, index);
  }

  [[nodiscard]] bool DoCommitLagrange(const Evals& evals,
                                      Commitment* out) const {
    return gwc_.DoCommitLagrange(evals, out);
  }

  [[nodiscard]] bool DoCommitLagrange(const Evals& evals,
                                      crypto::BatchCommitmentState& state,
                                      size_t index) {
    return gwc_.DoCommitLagrange(evals, state, index);
  }

  template <typename ScalarContainer>
  [[nodiscard]] bool DoCommitLagrange(const ScalarContainer& v,
                                      Commitment* out) const {
    return gwc_.DoCommitLagrange(v, out);
  }

  template <typename ScalarContainer>
  [[nodiscard]] bool DoCommitLagrange(const ScalarContainer& v,
                                      crypto::BatchCommitmentState& state,
                                      size_t index) {
    return gwc_.DoCommitLagrange(v, state, index);
  }

  template <typename Container, typename Proof>
  [[nodiscard]] bool DoCreateOpeningProof(const Container& poly_openings,
                                          Proof* proof) {
    return gwc_.DoCreateOpeningProof(poly_openings, proof);
  }

  template <typename Container, typename Proof>
  [[nodiscard]] bool DoVerifyOpeningProof(const Container& poly_openings,
                                          Proof* proof) const {
    return gwc_.DoVerifyOpeningProof(poly_openings, proof);
  }

 private:
  template <typename PCS, typename LS>
  friend class c::zk::plonk::halo2::KZGFamilyProverImpl;
  friend class halo2_api::bn254::Prover;
  friend class base::Copyable<
      GWCExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>>;

  const std::vector<G1Point, base::memory::ReusingAllocator<G1Point>>&
  GetG1PowersOfTau() const {
    return this->gwc_.kzg().g1_powers_of_tau();
  }

  const std::vector<G1Point, base::memory::ReusingAllocator<G1Point>>&
  GetG1PowersOfTauLagrange() const {
    return this->gwc_.kzg().g1_powers_of_tau_lagrange();
  }

  template <typename BaseContainer, typename ScalarContainer,
            typename OutCommitment>
  [[nodiscard]] bool DoMSM(const BaseContainer& bases,
                           const ScalarContainer& scalars,
                           OutCommitment* out) const {
    return this->gwc_.kzg().DoMSM(bases, scalars, out);
  }

  crypto::GWC<Curve, MaxDegree, Commitment> gwc_;
};

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename Commitment>
struct UnivariatePolynomialCommitmentSchemeExtensionTraits<
    GWCExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>> {
 public:
  constexpr static size_t kMaxExtendedDegree = MaxExtendedDegree;
  constexpr static size_t kMaxExtendedSize = kMaxExtendedDegree + 1;
};

}  // namespace zk

namespace crypto {

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename _Commitment>
struct VectorCommitmentSchemeTraits<
    zk::GWCExtension<Curve, MaxDegree, MaxExtendedDegree, _Commitment>> {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using Field = typename G1Point::ScalarField;
  using Commitment = _Commitment;

  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = false;
  constexpr static bool kSupportsBatchMode = true;
};

}  // namespace crypto

namespace base {

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree,
          typename Commitment>
class Copyable<
    zk::GWCExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>> {
 public:
  using PCS = zk::GWCExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>;

  static bool WriteTo(const PCS& pcs, Buffer* buffer) {
    return buffer->WriteMany(pcs.gwc_);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, PCS* pcs) {
    crypto::GWC<Curve, MaxDegree, Commitment> gwc;
    if (!buffer.ReadMany(&gwc)) {
      return false;
    }

    pcs->gwc_ = std::move(gwc);
    return true;
  }

  static size_t EstimateSize(const PCS& pcs) {
    return base::EstimateSize(pcs.gwc_);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_BASE_COMMITMENTS_GWC_EXTENSION_H_
