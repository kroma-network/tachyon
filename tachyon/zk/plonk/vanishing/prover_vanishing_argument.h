// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_PROVER_VANISHING_ARGUMENT_H_
#define TACHYON_ZK_PLONK_VANISHING_PROVER_VANISHING_ARGUMENT_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "tachyon/base/parallelize.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/vanishing/vanishing_committed.h"
#include "tachyon/zk/plonk/vanishing/vanishing_constructed.h"
#include "tachyon/zk/plonk/vanishing/vanishing_evaluated.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"

namespace tachyon::zk::plonk {

template <typename PCS, typename Poly = typename PCS::Poly>
VanishingCommitted<Poly> CommitRandomPoly(ProverBase<PCS>* prover) {
  using F = typename PCS::Field;

  // Sample a random polynomial of degree n - 1
  // TODO(TomTaehoonKim): Figure out why it is named |random_poly|.
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/vanishing/prover.rs#L52-L54
  Poly random_poly = Poly::One();

  // Sample a random blinding factor
  // TODO(TomTaehoonKim): Figure out why it is named |random_blind|.
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/vanishing/prover.rs#L55-L56
  F random_blind = F::Zero();

  prover->CommitAndWriteToProof(random_poly);

  return {std::move(random_poly), std::move(random_blind)};
}

template <typename PCS, typename Poly, typename F, typename C,
          typename ExtendedEvals,
          typename ExtendedPoly = typename PCS::ExtendedPoly>
VanishingConstructed<Poly, ExtendedPoly> CommitFinalHPoly(
    ProverBase<PCS>* prover, VanishingCommitted<Poly>&& committed,
    const VerifyingKey<F, C>& vk, ExtendedEvals& circuit_column) {
  // Divide by t(X) = X^{params.n} - 1.
  ExtendedEvals& h_evals = DivideByVanishingPolyInPlace<F>(
      circuit_column, prover->extended_domain(), prover->domain());

  // Obtain final h(X) polynomial
  ExtendedPoly h_poly =
      ExtendedToCoeff<F, ExtendedPoly>(h_evals, prover->extended_domain());

  // Truncate it to match the size of the quotient polynomial; the
  // evaluation domain might be slightly larger than necessary because
  // it always lies on a power-of-two boundary.
  std::vector<F>& h_coeffs = h_poly.coefficients().coefficients();
  const size_t quotient_poly_degree =
      vk.constraint_system().ComputeDegree() - 1;
  size_t n = prover->pcs().N();
  h_coeffs.resize(n * quotient_poly_degree, F::Zero());

  // Compute commitments to each h(X) piece
  if constexpr (PCS::kSupportsBatchMode) {
    prover->pcs().SetBatchMode(quotient_poly_degree);
    base::ParallelizeByChunkSize(
        h_coeffs, n, [prover](absl::Span<const F> h_piece, size_t chunk_index) {
          prover->BatchCommitAt(h_piece, chunk_index);
        });
    prover->RetrieveAndWriteBatchCommitmentsToProof();
  } else {
    std::vector<C> commitments = base::ParallelizeMapByChunkSize(
        h_coeffs, n, [prover](absl::Span<const F> h_piece) {
          return prover->Commit(h_piece);
        });
    for (const C& commitment : commitments) {
      CHECK(prover->GetWriter()->WriteToProof(commitment));
    }
  }

  // FIXME(TomTaehoonKim): Remove this if possible.
  std::vector<F> h_blinds =
      base::CreateVector(quotient_poly_degree,
                         [prover]() { return prover->blinder().Generate(); });

  return {std::move(h_poly), std::move(h_blinds), std::move(committed)};
}

template <typename PCS, typename Poly, typename ExtendedPoly, typename F,
          typename Commitment>
VanishingEvaluated<Poly> CommitRandomEval(
    const PCS& pcs, VanishingConstructed<Poly, ExtendedPoly>&& constructed,
    const F& x, const F& x_n, crypto::TranscriptWriter<Commitment>* writer) {
  using Coeffs = typename Poly::Coefficients;

  size_t n = pcs.N();
  auto h_chunks =
      base::Chunked(constructed.h_poly().coefficients().coefficients(), n);
  std::vector<absl::Span<F>> h_pieces =
      base::Map(h_chunks.begin(), h_chunks.end(),
                [](absl::Span<F> h_piece) { return h_piece; });
  std::vector<F> coeffs = base::CreateVector(n, F::Zero());
  for (size_t i = h_pieces.size() - 1; i != SIZE_MAX; --i) {
    OPENMP_PARALLEL_FOR(size_t j = 0; j < n; ++j) {
      coeffs[j] *= x_n;
      coeffs[j] += h_pieces[i][j];
    }
  }

  Poly h_poly(Coeffs(std::move(coeffs)));
  F h_blind = Poly(Coeffs(constructed.h_blinds())).Evaluate(x_n);

  VanishingCommitted<Poly> committed = std::move(constructed).TakeCommitted();
  F random_eval = committed.random_poly().Evaluate(x);
  CHECK(writer->WriteToProof(random_eval));

  return {std::move(h_poly), std::move(h_blind), std::move(committed)};
}

template <typename Poly, typename F>
std::vector<crypto::PolynomialOpening<Poly>> OpenVanishingArgument(
    const VanishingEvaluated<Poly>& evaluated, const F& x) {
  base::DeepRef<const F> x_ref(&x);
  return {crypto::PolynomialOpening<Poly>(
              base::Ref<const Poly>(&evaluated.h_poly()), x_ref,
              evaluated.h_poly().Evaluate(x)),
          crypto::PolynomialOpening<Poly>(
              base::Ref<const Poly>(&evaluated.committed().random_poly()),
              x_ref, evaluated.committed().random_poly().Evaluate(x))};
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_PROVER_VANISHING_ARGUMENT_H_
