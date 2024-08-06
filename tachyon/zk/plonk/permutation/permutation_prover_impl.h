// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVER_IMPL_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVER_IMPL_H_

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/base/profiler.h"
#include "tachyon/base/ref.h"
#include "tachyon/zk/plonk/permutation/grand_product_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_prover.h"
#include "tachyon/zk/plonk/permutation/permutation_utils.h"

namespace tachyon::zk::plonk {

template <typename Poly, typename Evals>
template <typename PCS>
void PermutationProver<Poly, Evals>::CreateGrandProductPolys(
    ProverBase<PCS>* prover, const PermutationTableStore<Evals>& table_store,
    size_t chunk_num, const F& beta, const F& gamma) {
  TRACE_EVENT("Utils", "CreateGrandProductPolys");
  grand_product_polys_.reserve(chunk_num);

  // Track the "last" value from the previous column set.
  F last_z = F::One();

  for (size_t i = 0; i < chunk_num; ++i) {
    std::vector<base::Ref<const Evals>> permuted_columns =
        table_store.GetPermutedColumns(i);
    std::vector<base::Ref<const Evals>> unpermuted_columns =
        table_store.GetUnpermutedColumns(i);
    std::vector<base::Ref<const Evals>> value_columns =
        table_store.GetValueColumns(i);

    size_t chunk_size = table_store.GetChunkSize(i);
    Evals grand_product_poly = GrandProductArgument::CreateExcessivePoly(
        prover,
        CreateNumeratorCallback(unpermuted_columns, value_columns, beta, gamma),
        CreateDenominatorCallback(permuted_columns, value_columns, beta, gamma),
        chunk_size, last_z);

    grand_product_polys_.emplace_back(std::move(grand_product_poly),
                                      prover->blinder().Generate());
  }
}

// static
template <typename Poly, typename Evals>
template <typename PCS, typename ExtendedEvals>
void PermutationProver<Poly, Evals>::BatchCreateGrandProductPolys(
    std::vector<PermutationProver>& permutation_provers,
    ProverBase<PCS>* prover, const PermutationArgument& argument,
    const std::vector<MultiPhaseRefTable<Evals>>& tables,
    size_t constraint_system_degree,
    const PermutationProvingKey<Poly, Evals, ExtendedEvals>&
        permutation_proving_key,
    const F& beta, const F& gamma) {
  TRACE_EVENT("ProofGeneration",
              "Plonk::Permutation::Prover::BatchCreateGrandProductPolys");
  // How many columns can be included in a single permutation polynomial?
  // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
  // will never underflow because of the requirement of at least a degree
  // 3 circuit for the permutation argument.
  CHECK_GE(constraint_system_degree, argument.RequiredDegree());
  if (argument.columns().empty()) return;

  size_t chunk_len = ComputePermutationChunkLength(constraint_system_degree);
  size_t chunk_num = (argument.columns().size() + chunk_len - 1) / chunk_len;

  UnpermutedTable<Evals> unpermuted_table = UnpermutedTable<Evals>::Construct(
      argument.columns().size(), prover->pcs().N(), prover->domain());
  PermutedTable<Evals> permuted_table(&permutation_proving_key.permutations());
  for (size_t i = 0; i < tables.size(); ++i) {
    PermutationTableStore<Evals> table_store(argument.columns(), tables[i],
                                             permuted_table, unpermuted_table,
                                             chunk_len);
    permutation_provers[i].CreateGrandProductPolys(prover, table_store,
                                                   chunk_num, beta, gamma);
  }
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
void PermutationProver<Poly, Evals>::BatchCommitGrandProductPolys(
    const std::vector<PermutationProver>& permutation_provers,
    ProverBase<PCS>* prover, size_t& commit_idx) {
  TRACE_EVENT("ProofGeneration",
              "Plonk::Permutation::Prover::BatchCommitGrandProductPolys");
  if (permutation_provers.empty()) return;

  if constexpr (PCS::kSupportsBatchMode) {
    for (const PermutationProver& permutation_prover : permutation_provers) {
      for (const BlindedPolynomial<Poly, Evals>& grand_product_poly :
           permutation_prover.grand_product_polys_) {
        prover->BatchCommitAt(grand_product_poly.evals(), commit_idx++);
      }
    }
  } else {
    for (const PermutationProver& permutation_prover : permutation_provers) {
      for (const BlindedPolynomial<Poly, Evals>& grand_product_poly :
           permutation_prover.grand_product_polys_) {
        prover->CommitAndWriteToProof(grand_product_poly.evals());
      }
    }
  }
}

template <typename Poly, typename Evals>
template <typename Domain>
void PermutationProver<Poly, Evals>::TransformEvalsToPoly(
    const Domain* domain) {
  TRACE_EVENT("Utils", "TransformEvalsToPoly");
  for (BlindedPolynomial<Poly, Evals>& grand_product_poly :
       grand_product_polys_) {
    grand_product_poly.TransformEvalsToPoly(domain);
  }
}

template <typename Poly, typename Evals>
template <typename PCS>
void PermutationProver<Poly, Evals>::Evaluate(
    ProverBase<PCS>* prover,
    const PermutationOpeningPointSet<F>& point_set) const {
  TRACE_EVENT("Utils", "Evaluate");
  // THE ORDER IS IMPORTANT!! DO NOT CHANGE!
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a369/halo2_proofs/src/plonk/permutation/prover.rs#L231-L276.
  for (size_t i = 0; i < grand_product_polys_.size(); ++i) {
    const Poly& poly = grand_product_polys_[i].poly();
    // Zₚ,ᵢ(x)
    prover->EvaluateAndWriteToProof(poly, point_set.x);
    // Zₚ,ᵢ(ω * x)
    prover->EvaluateAndWriteToProof(poly, point_set.x_next);
    // If we have any remaining sets to process, evaluate this set at ωᵘ
    // so we can constrain the last value of its running product to equal the
    // first value of the next set's running product, chaining them together.
    if (i != grand_product_polys_.size() - 1) {
      // Zₚ,ᵢ(ω^(last) * x)
      prover->EvaluateAndWriteToProof(poly, point_set.x_last);
    }
  }
}

// static
template <typename Poly, typename Evals>
template <typename PCS, typename ExtendedEvals>
void PermutationProver<Poly, Evals>::EvaluateProvingKey(
    ProverBase<PCS>* prover,
    const PermutationProvingKey<Poly, Evals, ExtendedEvals>& proving_key,
    const PermutationOpeningPointSet<F>& point_set) {
  TRACE_EVENT("ProofGeneration",
              "Plonk::Permutation::Prover::EvaluateProvingKey");
  for (const Poly& poly : proving_key.polys()) {
    prover->EvaluateAndWriteToProof(poly, point_set.x);
  }
}

template <typename Poly, typename Evals>
void PermutationProver<Poly, Evals>::Open(
    const PermutationOpeningPointSet<F>& point_set,
    std::vector<crypto::PolynomialOpening<Poly>>& openings) const {
  TRACE_EVENT("ProofGeneration", "Plonk::Permutation::Prover::Open");
  if (grand_product_polys_.empty()) return;

#define OPENING(poly, point) \
  base::Ref<const Poly>(&poly), point_set.point, poly.Evaluate(point_set.point)

  // THE ORDER IS IMPORTANT!! DO NOT CHANGE!
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a369/halo2_proofs/src/plonk/permutation/prover.rs#L279-L324.
  for (const BlindedPolynomial<Poly, Evals>& grand_product_poly :
       grand_product_polys_) {
    const Poly& poly = grand_product_poly.poly();
    // Zₚ,ᵢ(x)
    openings.emplace_back(OPENING(poly, x));
    // Zₚ,ᵢ(ω * x)
    openings.emplace_back(OPENING(poly, x_next));
  }

  for (auto it = grand_product_polys_.rbegin() + 1;
       it != grand_product_polys_.rend(); ++it) {
    const Poly& poly = it->poly();
    // Zₚ,ᵢ(ω^(last) * x)
    openings.emplace_back(OPENING(poly, x_last));
  }
#undef OPENING
}

// static
template <typename Poly, typename Evals>
template <typename ExtendedEvals>
void PermutationProver<Poly, Evals>::OpenPermutationProvingKey(
    const PermutationProvingKey<Poly, Evals, ExtendedEvals>& proving_key,
    const PermutationOpeningPointSet<F>& point_set,
    std::vector<crypto::PolynomialOpening<Poly>>& openings) {
  TRACE_EVENT("ProofGeneration",
              "Plonk::Permutation::Prover::OpenPermutationProvingKey");
#define OPENING(poly, point) \
  base::Ref<const Poly>(&poly), point_set.point, poly.Evaluate(point_set.point)

  for (const Poly& poly : proving_key.polys()) {
    openings.emplace_back(OPENING(poly, x));
  }
#undef OPENING
}

// static
template <typename Poly, typename Evals>
std::function<typename Poly::Field(size_t, RowIndex)>
PermutationProver<Poly, Evals>::CreateNumeratorCallback(
    const std::vector<base::Ref<const Evals>>& unpermuted_columns,
    const std::vector<base::Ref<const Evals>>& value_columns, const F& beta,
    const F& gamma) {
  // vᵢ(ωʲ) + β * δⁱ * ωʲ + γ
  return [&unpermuted_columns, &value_columns, &beta, &gamma](
             size_t column_index, RowIndex row_index) {
    const Evals& unpermuted_values = *unpermuted_columns[column_index];
    const Evals& values = *value_columns[column_index];
    return values[row_index] + beta * unpermuted_values[row_index] + gamma;
  };
}

// static
template <typename Poly, typename Evals>
std::function<typename Poly::Field(size_t, RowIndex)>
PermutationProver<Poly, Evals>::CreateDenominatorCallback(
    const std::vector<base::Ref<const Evals>>& permuted_columns,
    const std::vector<base::Ref<const Evals>>& value_columns, const F& beta,
    const F& gamma) {
  // vᵢ(ωʲ) + β * sᵢ(ωʲ) + γ
  return [&permuted_columns, &value_columns, &beta, &gamma](
             size_t column_index, RowIndex row_index) {
    const Evals& permuted_values = *permuted_columns[column_index];
    const Evals& values = *value_columns[column_index];
    return values[row_index] + beta * permuted_values[row_index] + gamma;
  };
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVER_IMPL_H_
