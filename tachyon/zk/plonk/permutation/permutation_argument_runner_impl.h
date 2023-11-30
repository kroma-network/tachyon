// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_RUNNER_IMPL_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_RUNNER_IMPL_H_

#include <functional>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/plonk/permutation/grand_product_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_argument_runner.h"
#include "tachyon/zk/plonk/permutation/permutation_table_store.h"
#include "tachyon/zk/plonk/permutation/permuted_table.h"
#include "tachyon/zk/plonk/permutation/unpermuted_table.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
template <typename PCSTy, typename F>
PermutationCommitted<Poly>
PermutationArgumentRunner<Poly, Evals>::CommitArgument(
    Prover<PCSTy>* prover, const PermutationArgument& argument,
    Table<Evals>& table, size_t constraint_system_degree,
    const PermutationProvingKey<PCSTy>& permutation_proving_key, const F& beta,
    const F& gamma) {
  // How many columns can be included in a single permutation polynomial?
  // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
  // will never underflow because of the requirement of at least a degree
  // 3 circuit for the permutation argument.
  CHECK_GE(constraint_system_degree, argument.RequiredDegree());

  size_t chunk_size = constraint_system_degree - 2;
  size_t chunk_num = (argument.columns().size() + chunk_size - 1) / chunk_size;

  UnpermutedTable<Evals> unpermuted_table = UnpermutedTable<Evals>::Construct(
      argument.columns().size(), prover->domain());
  PermutedTable<Evals> permuted_table(&permutation_proving_key.permutations());
  PermutationTableStore<Evals> table_store(
      argument.columns(), table, permuted_table, unpermuted_table, chunk_size);

  std::vector<BlindedPolynomial<Poly>> grand_product_polys;
  grand_product_polys.reserve(chunk_num);

  // Track the "last" value from the previous column set.
  F last_z = F::One();

  for (size_t i = 0; i < chunk_num; ++i) {
    std::vector<Ref<const Evals>> permuted_columns =
        table_store.GetPermutedColumns(i);
    std::vector<Ref<const Evals>> unpermuted_columns =
        table_store.GetUnpermutedColumns(i);
    std::vector<Ref<const Evals>> value_columns =
        table_store.GetValueColumns(i);

    size_t chunk_size = table_store.GetChunkSize(i);
    BlindedPolynomial<Poly> grand_product_poly =
        GrandProductArgument::CommitExcessive(
            prover,
            CreateNumeratorCallback<F>(unpermuted_columns, value_columns, beta,
                                       gamma),
            CreateDenominatorCallback<F>(permuted_columns, value_columns, beta,
                                         gamma),
            chunk_size, last_z);

    grand_product_polys.push_back(std::move(grand_product_poly));
  }

  return PermutationCommitted<Poly>(std::move(grand_product_polys));
}

template <typename Poly, typename Evals>
template <typename F>
std::function<base::ParallelizeCallback3<F>(size_t)>
PermutationArgumentRunner<Poly, Evals>::CreateNumeratorCallback(
    const std::vector<Ref<const Evals>>& unpermuted_columns,
    const std::vector<Ref<const Evals>>& value_columns, const F& beta,
    const F& gamma) {
  // vᵢ(ωʲ) + β * δⁱ * ωʲ + γ
  return [&unpermuted_columns, &value_columns, &beta,
          &gamma](size_t column_index) {
    const Evals& unpermuted_values = *unpermuted_columns[column_index];
    const Evals& values = *value_columns[column_index];
    return [&unpermuted_values, &values, &beta, &gamma](
               absl::Span<F> chunk, size_t chunk_index, size_t chunk_size_in) {
      size_t i = chunk_index * chunk_size_in;
      for (F& result : chunk) {
        result *= *values[i] + beta * *unpermuted_values[i] + gamma;
        ++i;
      }
    };
  };
}

template <typename Poly, typename Evals>
template <typename F>
std::function<base::ParallelizeCallback3<F>(size_t)>
PermutationArgumentRunner<Poly, Evals>::CreateDenominatorCallback(
    const std::vector<Ref<const Evals>>& permuted_columns,
    const std::vector<Ref<const Evals>>& value_columns, const F& beta,
    const F& gamma) {
  // vᵢ(ωʲ) + β * sᵢ(ωʲ) + γ
  return [&permuted_columns, &value_columns, &beta,
          &gamma](size_t column_index) {
    const Evals& permuted_values = *permuted_columns[column_index];
    const Evals& values = *value_columns[column_index];
    return [&permuted_values, &values, &beta, &gamma](
               absl::Span<F> chunk, size_t chunk_index, size_t chunk_size_in) {
      size_t i = chunk_index * chunk_size_in;
      for (F& result : chunk) {
        result *= *values[i] + beta * *permuted_values[i] + gamma;
        ++i;
      }
    };
  };
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_RUNNER_IMPL_H_
