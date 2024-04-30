// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_PROVER_IMPL_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_PROVER_IMPL_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/vanishing/vanishing_argument.h"
#include "tachyon/zk/plonk/vanishing/vanishing_prover.h"

namespace tachyon::zk::plonk {

template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename PCS>
void VanishingProver<Poly, Evals, ExtendedPoly,
                     ExtendedEvals>::CreateRandomPoly(ProverBase<PCS>* prover) {
  // Sample a random polynomial of degree n - 1.
  // TODO(TomTaehoonKim): Figure out why it is named |random_poly|.
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/vanishing/prover.rs#L52-L54
  Poly random_poly = Poly::One();

  // Sample a random blinding factor.
  // TODO(TomTaehoonKim): Figure out why it is named |random_blind|.
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/vanishing/prover.rs#L55-L56
  F random_blind = F::Zero();

  random_poly_ = {std::move(random_poly), std::move(random_blind)};
}

template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename PCS>
void VanishingProver<Poly, Evals, ExtendedPoly,
                     ExtendedEvals>::CommitRandomPoly(ProverBase<PCS>* prover,
                                                      size_t& commit_idx)
    const {
  if constexpr (PCS::kSupportsBatchMode) {
    prover->BatchCommitAt(random_poly_.poly(), commit_idx++);
  } else {
    prover->CommitAndWriteToProof(random_poly_.poly());
  }
}

template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename PCS, typename C>
void VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::CreateHEvals(
    ProverBase<PCS>* prover, const ProvingKey<Poly, Evals, C>& proving_key,
    const std::vector<MultiPhaseRefTable<Poly>>& tables, const F& theta,
    const F& beta, const F& gamma, const F& y,
    const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
    const std::vector<lookup::halo2::Prover<Poly, Evals>>& lookup_provers) {
  VanishingArgument<F, Evals> vanishing_argument =
      VanishingArgument<F, Evals>::Create(
          proving_key.verifying_key().constraint_system());
  F zeta = GetHalo2Zeta<F>();
  h_evals_ = vanishing_argument.BuildExtendedCircuitColumn(
      prover, proving_key, tables, theta, beta, gamma, y, zeta,
      permutation_provers, lookup_provers);
}

template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename PCS>
void VanishingProver<Poly, Evals, ExtendedPoly,
                     ExtendedEvals>::CreateFinalHPoly(ProverBase<PCS>* prover,
                                                      const ConstraintSystem<F>&
                                                          constraint_system) {
  // Divide by t(X) = Xⁿ - 1.
  DivideByVanishingPolyInPlace<F>(h_evals_, prover->extended_domain(),
                                  prover->domain());

  // Obtain final h(X) polynomial
  h_poly_ = ExtendedToCoeff<F, ExtendedPoly>(std::move(h_evals_),
                                             prover->extended_domain());

  // FIXME(TomTaehoonKim): Remove this if possible.
  const size_t quotient_poly_degree = constraint_system.ComputeDegree() - 1;
  h_blinds_ = base::CreateVector(quotient_poly_degree, [prover]() {
    return prover->blinder().Generate();
  });
}

template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename PCS>
void VanishingProver<Poly, Evals, ExtendedPoly,
                     ExtendedEvals>::CommitFinalHPoly(ProverBase<PCS>* prover,
                                                      const ConstraintSystem<F>&
                                                          constraint_system,
                                                      size_t& commit_idx) {
  // Truncate it to match the size of the quotient polynomial; the
  // evaluation domain might be slightly larger than necessary because
  // it always lies on a power-of-two boundary.
  std::vector<F>& h_coeffs = h_poly_.coefficients().coefficients();
  const size_t quotient_poly_degree = constraint_system.ComputeDegree() - 1;
  size_t n = prover->pcs().N();
  h_coeffs.resize(n * quotient_poly_degree, F::Zero());

  // Compute commitments to each h(X) piece
  if constexpr (PCS::kSupportsBatchMode) {
    base::ParallelizeByChunkSize(
        h_coeffs, n,
        [commit_idx, prover](absl::Span<const F> h_piece, size_t chunk_index) {
          prover->BatchCommitAt(h_piece, commit_idx + chunk_index);
        });
    commit_idx += quotient_poly_degree;
  } else {
    using Commitment = typename PCS::Commitment;
    std::vector<Commitment> commitments = base::ParallelizeMapByChunkSize(
        h_coeffs, n, [prover](absl::Span<const F> h_piece) {
          return prover->Commit(h_piece);
        });
    for (const Commitment& commitment : commitments) {
      CHECK(prover->GetWriter()->WriteToProof(commitment));
    }
  }
}

// static
template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename PCS, ColumnType C>
void VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::EvaluateColumns(
    ProverBase<PCS>* prover, const absl::Span<const Poly> polys,
    const std::vector<QueryData<C>>& queries, const F& x) {
  for (const QueryData<C>& query : queries) {
    const Poly& poly = polys[query.column().index()];
    prover->EvaluateAndWriteToProof(
        poly, query.rotation().RotateOmega(prover->domain(), x));
  }
}

template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename PCS>
void VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::BatchEvaluate(
    ProverBase<PCS>* prover, const ConstraintSystem<F>& constraint_system,
    const std::vector<MultiPhaseRefTable<Poly>>& tables, const F& x,
    const F& x_n) {
  using Coefficients = typename Poly::Coefficients;

  size_t num_circuits = tables.size();
  for (size_t i = 0; i < num_circuits; ++i) {
    if constexpr (PCS::kQueryInstance) {
      EvaluateColumns(prover, tables[i].GetInstanceColumns(),
                      constraint_system.instance_queries(), x);
    }
  }

  for (size_t i = 0; i < num_circuits; ++i) {
    EvaluateColumns(prover, tables[i].GetAdviceColumns(),
                    constraint_system.advice_queries(), x);
  }

  EvaluateColumns(prover, tables[0].GetFixedColumns(),
                  constraint_system.fixed_queries(), x);

  size_t n = prover->pcs().N();
  auto h_chunks = base::Chunked(h_poly_.coefficients().coefficients(), n);
  std::vector<absl::Span<F>> h_pieces =
      base::Map(h_chunks.begin(), h_chunks.end(),
                [](absl::Span<F> h_piece) { return h_piece; });
  std::vector<F> coeffs(n);
  for (size_t i = h_pieces.size() - 1; i != SIZE_MAX; --i) {
    OPENMP_PARALLEL_FOR(size_t j = 0; j < n; ++j) {
      coeffs[j] *= x_n;
      coeffs[j] += h_pieces[i][j];
    }
  }
  combined_h_poly_ = Poly(Coefficients(std::move(coeffs)));

  prover->EvaluateAndWriteToProof(random_poly_.poly(), x);
}

// static
template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename Domain, ColumnType C>
void VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::OpenColumns(
    const Domain* domain, const absl::Span<const Poly> polys,
    const std::vector<QueryData<C>>& queries, const F& x,
    std::vector<crypto::PolynomialOpening<Poly>>& openings) {
#define OPENING(poly, point) \
  base::Ref<const Poly>(&poly), point, poly.Evaluate(point)

  for (const QueryData<C>& query : queries) {
    const Poly& poly = polys[query.column().index()];
    F point = query.rotation().RotateOmega(domain, x);
    openings.emplace_back(OPENING(poly, point));
  }
#undef OPENING
}

// static
template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename PCS, typename Domain>
void VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::
    OpenAdviceInstanceColumns(
        const Domain* domain, const ConstraintSystem<F>& constraint_system,
        const MultiPhaseRefTable<Poly>& table, const F& x,
        std::vector<crypto::PolynomialOpening<Poly>>& openings) {
  if constexpr (PCS::kQueryInstance) {
    OpenColumns(domain, table.GetInstanceColumns(),
                constraint_system.instance_queries(), x, openings);
  }
  OpenColumns(domain, table.GetAdviceColumns(),
              constraint_system.advice_queries(), x, openings);
}

// static
template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
template <typename Domain>
void VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::
    OpenFixedColumns(const Domain* domain,
                     const ConstraintSystem<F>& constraint_system,
                     const MultiPhaseRefTable<Poly>& table, const F& x,
                     std::vector<crypto::PolynomialOpening<Poly>>& openings) {
  OpenColumns(domain, table.GetFixedColumns(),
              constraint_system.fixed_queries(), x, openings);
}

template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
void VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::Open(
    const F& x, std::vector<crypto::PolynomialOpening<Poly>>& openings) const {
#define OPENING(poly, point) \
  base::Ref<const Poly>(&poly), point, poly.Evaluate(point)

  openings.emplace_back(OPENING(combined_h_poly_, x));
  openings.emplace_back(OPENING(random_poly_.poly(), x));
#undef OPENING
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_PROVER_IMPL_H_
