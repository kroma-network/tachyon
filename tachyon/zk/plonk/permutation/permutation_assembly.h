// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ASSEMBLY_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ASSEMBLY_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/export.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/permutation/cycle_store.h"
#include "tachyon/zk/plonk/permutation/label.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"
#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"
#include "tachyon/zk/plonk/permutation/unpermuted_table.h"

namespace tachyon::zk::plonk {

// Struct that accumulates all the necessary data in order to construct the
// permutation argument.
class TACHYON_EXPORT PermutationAssembly {
 public:
  PermutationAssembly() = default;

  PermutationAssembly(const PermutationArgument& p, RowIndex rows)
      : PermutationAssembly(p.columns(), rows) {}

  PermutationAssembly(const std::vector<AnyColumnKey>& columns, RowIndex rows)
      : columns_(columns),
        cycle_store_(CycleStore(columns_.size(), rows)),
        rows_(rows) {}

  PermutationAssembly(std::vector<AnyColumnKey>&& columns, RowIndex rows)
      : columns_(std::move(columns)),
        cycle_store_(CycleStore(columns_.size(), rows)),
        rows_(rows) {}

  static PermutationAssembly CreateForTesting(std::vector<AnyColumnKey> columns,
                                              CycleStore cycle_store,
                                              RowIndex rows) {
    PermutationAssembly ret;
    ret.columns_ = std::move(columns);
    ret.cycle_store_ = std::move(cycle_store);
    ret.rows_ = rows;
    return ret;
  }

  const std::vector<AnyColumnKey>& columns() const { return columns_; }
  const CycleStore& cycle_store() const { return cycle_store_; }

  void Copy(const AnyColumnKey& left_column, RowIndex left_row,
            const AnyColumnKey& right_column, RowIndex right_row) {
    CHECK_LE(left_row, rows_);
    CHECK_LE(right_row, rows_);

    // Get indices of each column.
    size_t left_col_idx = GetColumnIndex(left_column);
    size_t right_col_idx = GetColumnIndex(right_column);

    cycle_store_.MergeCycle(Label(left_col_idx, left_row),
                            Label(right_col_idx, right_row));
  }

  // Returns |PermutationVerifyingKey| which has commitments for permutations.
  template <typename PCS, typename Evals,
            typename Commitment = typename PCS::Commitment>
  constexpr PermutationVerifyingKey<Commitment> BuildVerifyingKey(
      const Entity<PCS>* entity, const std::vector<Evals>& permutations) const {
    const PCS& pcs = entity->pcs();
    return PermutationVerifyingKey<Commitment>(
        base::Map(permutations, [&pcs](const Evals& permutation) {
          Commitment commitment;
          CHECK(pcs.CommitLagrange(permutation, &commitment));
          return commitment;
        }));
  }

  // Returns the |PermutationProvingKey| that has the coefficient form and
  // evaluation form of the permutation.
  template <typename PCS, typename Poly = typename PCS::Poly,
            typename Evals = typename PCS::Evals>
  constexpr PermutationProvingKey<Poly, Evals> BuildProvingKey(
      const ProverBase<PCS>* prover,
      const std::vector<Evals>& permutations) const {
    using Domain = typename PCS::Domain;

    const Domain* domain = prover->domain();

    // The polynomials of permutations with coefficients.
    std::vector<Poly> polys;
    polys.reserve(columns_.size());
    for (size_t i = 0; i < columns_.size(); ++i) {
      Poly poly = domain->IFFT(permutations[i]);
      polys.push_back(std::move(poly));
    }

    return PermutationProvingKey<Poly, Evals>(std::move(permutations),
                                              std::move(polys));
  }

  // Generate the permutation polynomials based on the accumulated copy
  // permutations. Note that the permutation polynomials are in evaluation
  // form.
  template <typename Evals, typename Domain>
  std::vector<Evals> GeneratePermutations(const Domain* domain) const {
    CHECK_EQ(domain->size(), size_t{rows_});
    UnpermutedTable<Evals> unpermuted_table =
        UnpermutedTable<Evals>::Construct(columns_.size(), rows_, domain);

    // Init evaluation formed polynomials with all-zero coefficients.
    std::vector<Evals> permutations =
        base::CreateVector(columns_.size(), domain->template Empty<Evals>());

    // Assign |unpermuted_table| to |permutations|.
    base::Parallelize(permutations, [&unpermuted_table, this](
                                        absl::Span<Evals> chunk, size_t c,
                                        size_t chunk_size) {
      size_t i = c * chunk_size;
      for (Evals& evals : chunk) {
        for (size_t j = 0; j < rows_; ++j) {
          *evals[j] = unpermuted_table[cycle_store_.GetNextLabel(Label(i, j))];
        }
        ++i;
      }
    });
    return permutations;
  }

 private:
  size_t GetColumnIndex(const AnyColumnKey& column) const {
    return base::FindIndex(columns_, column).value();
  }

  // Columns that participate on the copy permutation argument.
  std::vector<AnyColumnKey> columns_;
  CycleStore cycle_store_;
  RowIndex rows_ = 0;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ASSEMBLY_H_
