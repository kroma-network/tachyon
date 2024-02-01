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

template <typename PCS, typename Poly>
[[nodiscard]] bool CommitRandomPoly(ProverBase<PCS>* prover,
                                    VanishingCommitted<Poly>* out) {
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

  *out = {std::move(random_poly), std::move(random_blind)};
  return true;
}

template <typename PCS, typename Poly, typename F, typename C,
          typename ExtendedEvals>
[[nodiscard]] bool CommitFinalHPoly(
    ProverBase<PCS>* prover, VanishingCommitted<Poly>&& committed,
    const VerifyingKey<F, C>& vk, ExtendedEvals& circuit_column,
    VanishingConstructed<Poly>* constructed_out) {
  using Coeffs = typename Poly::Coefficients;
  using ExtendedPoly = typename PCS::ExtendedPoly;

  // Divide by t(X) = X^{params.n} - 1.
  ExtendedEvals h_evals = DivideByVanishingPolyInPlace<F>(
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
  h_coeffs.resize(prover->pcs().N() * quotient_poly_degree, F::Zero());

  auto h_chunks = base::Chunked(h_coeffs, prover->pcs().N());
  std::vector<Poly> h_pieces = base::Map(
      h_chunks.begin(), h_chunks.end(), [](absl::Span<const F> h_piece) {
        return Poly(
            Coeffs(std::move(std::vector<F>(h_piece.begin(), h_piece.end()))));
      });

  // Compute commitments to each h(X) piece
  if constexpr (PCS::kSupportsBatchMode) {
    prover->pcs().SetBatchMode(h_pieces.size());
    base::ParallelizeByChunkSize(
        h_coeffs, prover->pcs().N(),
        [prover](absl::Span<const F> h_piece, size_t chunk_index) {
          prover->BatchCommitAt(h_piece, chunk_index);
        });
    prover->RetrieveAndWriteBatchCommitmentsToProof();
  } else {
    std::vector<C> commitments = base::ParallelizeMapByChunkSize(
        h_coeffs, prover->pcs().N(), [prover](absl::Span<const F> h_piece) {
          return prover->Commit(h_piece);
        });
    for (const C& commitment : commitments) {
      if (!prover->GetWriter()->WriteToProof(commitment)) return false;
    }
  }

  // FIXME(TomTaehoonKim): Remove this if possible.
  std::vector<F> h_blinds =
      base::CreateVector(quotient_poly_degree,
                         [prover]() { return prover->blinder().Generate(); });

  *constructed_out = {std::move(h_pieces), std::move(h_blinds),
                      std::move(committed)};
  return true;
}

template <typename PCS, typename Poly, typename F, typename Commitment>
[[nodiscard]] bool CommitRandomEval(
    const PCS& pcs, VanishingConstructed<Poly>&& constructed, const F& x,
    const F& x_n, crypto::TranscriptWriter<Commitment>* writer,
    VanishingEvaluated<Poly>* evaluated_out) {
  using Coeffs = typename Poly::Coefficients;

  Poly h_poly = Poly::template LinearCombination</*forward=*/false>(
      constructed.h_pieces(), x_n);

  F h_blind = Poly(Coeffs(constructed.h_blinds())).Evaluate(x_n);

  VanishingCommitted<Poly> committed = std::move(constructed).TakeCommitted();
  F random_eval = committed.random_poly().Evaluate(x);
  if (!writer->WriteToProof(random_eval)) return false;

  *evaluated_out = {std::move(h_poly), std::move(h_blind),
                    std::move(committed)};
  return true;
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
