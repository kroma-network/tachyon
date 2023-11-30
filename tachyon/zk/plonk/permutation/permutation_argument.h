// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/contains.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/prover.h"
#include "tachyon/zk/plonk/circuit/column_key.h"
#include "tachyon/zk/plonk/permutation/grand_product_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_committed.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"
#include "tachyon/zk/plonk/permutation/permutation_table_store.h"

namespace tachyon::zk {

// TODO(dongchangYoo): impl logics related to prove and verify.
class PermutationArgument {
 public:
  PermutationArgument() = default;
  explicit PermutationArgument(const std::vector<AnyColumnKey>& columns)
      : columns_(columns) {}
  explicit PermutationArgument(std::vector<AnyColumnKey>&& columns)
      : columns_(std::move(columns)) {}

  const std::vector<AnyColumnKey>& columns() const { return columns_; }

  void AddColumn(const AnyColumnKey& column) {
    if (base::Contains(columns_, column)) return;
    columns_.push_back(column);
  }

  // Returns the minimum circuit degree required by the permutation argument.
  // The argument may use larger degree gates depending on the actual
  // circuit's degree and how many columns are involved in the permutation.
  size_t RequiredDegree() const {
    // degree 2:
    // l₀(X) * (1 - z(X)) = 0
    //
    // We will fit as many polynomials pᵢ(X) as possible
    // into the required degree of the circuit, so the
    // following will not affect the required degree of
    // this middleware.
    //
    // clang-format off
    // (1 - (l_last(X) + l_blind(X))) * (z(ω * X) Π (p(X) + β * sᵢ(X) + γ) - z(X) Π (p(X) + β * δⁱ * X + γ))
    // clang-format on
    //
    // On the first sets of columns, except the first
    // set, we will do
    //
    // l₀(X) * (z(X) - z'(ω^(last) X)) = 0
    //
    // where z'(X) is the permutation for the previous set
    // of columns.
    //
    // On the final set of columns, we will do
    //
    // degree 3:
    // l_last(X) * (z'(X)² - z'(X)) = 0
    //
    // which will allow the last value to be zero to
    // ensure the argument is perfectly complete.

    // There are constraints of degree 3 regardless of the
    // number of columns involved.
    return 3;
  }

  // Returns commitments of Zₚ,ᵢ for chunk index i.
  //
  // See Halo2 book to figure out logic in detail.
  // https://zcash.github.io/halo2/design/proving-system/permutation.html
  template <typename PCSTy, typename ExtendedDomain,
            typename Evals = typename PCSTy::Evals,
            typename Poly = typename PCSTy::Poly,
            typename F = typename PCSTy::Field>
  PermutationCommitted<Poly> Commit(
      Prover<PCSTy, ExtendedDomain>* prover, Table<Evals>& table,
      size_t constraint_system_degree,
      const PermutationProvingKey<PCSTy>& permutation_proving_key,
      const F& beta, const F& gamma) {
    // How many columns can be included in a single permutation polynomial?
    // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
    // will never underflow because of the requirement of at least a degree
    // 3 circuit for the permutation argument.
    CHECK_GE(constraint_system_degree, RequiredDegree());

    size_t chunk_size = constraint_system_degree - 2;
    size_t chunk_num = (columns_.size() + chunk_size - 1) / chunk_size;

    UnpermutedTable<Evals> unpermuted_table =
        UnpermutedTable<Evals>::Construct(columns_.size(), prover->domain());
    PermutedTable<Evals> permuted_table(
        &permutation_proving_key.permutations());
    PermutationTableStore<Evals> table_store(columns_, table, permuted_table,
                                             unpermuted_table, chunk_size);

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
              CreateNumeratorCallback<Evals>(unpermuted_columns, value_columns,
                                             beta, gamma),
              CreateDenominatorCallback<Evals>(permuted_columns, value_columns,
                                               beta, gamma),
              chunk_size, last_z);

      grand_product_polys.push_back(std::move(grand_product_poly));
    }

    return PermutationCommitted<Poly>(std::move(grand_product_polys));
  }

 private:
  template <typename Evals, typename F = typename Evals::Field>
  std::function<base::ParallelizeCallback3<F>(size_t)> CreateNumeratorCallback(
      const std::vector<Ref<const Evals>>& unpermuted_columns,
      const std::vector<Ref<const Evals>>& value_columns, const F& beta,
      const F& gamma) {
    // vᵢ(ωʲ) + β * δⁱ * ωʲ + γ
    return [&unpermuted_columns, &value_columns, &beta,
            &gamma](size_t column_index) {
      const Evals& unpermuted_values = *unpermuted_columns[column_index];
      const Evals& values = *value_columns[column_index];
      return
          [&unpermuted_values, &values, &beta, &gamma](
              absl::Span<F> chunk, size_t chunk_index, size_t chunk_size_in) {
            size_t i = chunk_index * chunk_size_in;
            for (F& result : chunk) {
              result *= *values[i] + beta * *unpermuted_values[i] + gamma;
              ++i;
            }
          };
    };
  }

  template <typename Evals, typename F = typename Evals::Field>
  std::function<base::ParallelizeCallback3<F>(size_t)>
  CreateDenominatorCallback(
      const std::vector<Ref<const Evals>>& permuted_columns,
      const std::vector<Ref<const Evals>>& value_columns, const F& beta,
      const F& gamma) {
    // vᵢ(ωʲ) + β * sᵢ(ωʲ) + γ
    return [&permuted_columns, &value_columns, &beta,
            &gamma](size_t column_index) {
      const Evals& permuted_values = *permuted_columns[column_index];
      const Evals& values = *value_columns[column_index];
      return
          [&permuted_values, &values, &beta, &gamma](
              absl::Span<F> chunk, size_t chunk_index, size_t chunk_size_in) {
            size_t i = chunk_index * chunk_size_in;
            for (F& result : chunk) {
              result *= *values[i] + beta * *permuted_values[i] + gamma;
              ++i;
            }
          };
    };
  }

  std::vector<AnyColumnKey> columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_
