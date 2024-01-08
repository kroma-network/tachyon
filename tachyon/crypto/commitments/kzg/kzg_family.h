// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_FAMILY_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_FAMILY_H_

#include <utility>

#include "tachyon/crypto/commitments/kzg/kzg.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::crypto {

template <typename G1Point, size_t MaxDegree, typename Commitment>
class KZGFamily {
 public:
  using F = typename G1Point::ScalarField;

  KZGFamily() = default;
  explicit KZGFamily(KZG<G1Point, MaxDegree, Commitment>&& kzg)
      : kzg_(std::move(kzg)) {}

  size_t N() const { return kzg_.N(); }

  [[nodiscard]] bool DoUnsafeSetup(size_t size) {
    return DoUnsafeSetup(size, F::Random());
  }

  [[nodiscard]] bool DoUnsafeSetup(size_t size, const F& tau) {
    return kzg_.UnsafeSetup(size, tau) && DoUnsafeSetupWithTau(size, tau);
  }

  template <typename Container>
  [[nodiscard]] bool DoCommit(const Container& poly,
                              Commitment* commitment) const {
    return kzg_.Commit(poly, commitment);
  }

  template <typename Container>
  [[nodiscard]] bool DoCommitLagrange(const Container& poly,
                                      Commitment* commitment) const {
    return kzg_.CommitLagrange(poly, commitment);
  }

  [[nodiscard]] bool DoCommit(
      const math::UnivariateDensePolynomial<F, MaxDegree>& poly,
      Commitment* commitment) const {
    return kzg_.Commit(poly.coefficients().coefficients(), commitment);
  }

  [[nodiscard]] bool DoCommitLagrange(
      const math::UnivariateEvaluations<F, MaxDegree>& evals,
      Commitment* commitment) const {
    return kzg_.CommitLagrange(evals.evaluations(), commitment);
  }

 protected:
  [[nodiscard]] virtual bool DoUnsafeSetupWithTau(size_t size,
                                                  const F& tau) = 0;

  KZG<G1Point, MaxDegree, Commitment> kzg_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_FAMILY_H_
