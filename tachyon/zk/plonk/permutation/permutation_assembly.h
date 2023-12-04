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
#include "tachyon/zk/base/entities/prover.h"
#include "tachyon/zk/plonk/permutation/cycle_store.h"
#include "tachyon/zk/plonk/permutation/label.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"
#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"
#include "tachyon/zk/plonk/permutation/unpermuted_table.h"

namespace tachyon::zk {

// Struct that accumulates all the necessary data in order to construct the
// permutation argument.
template <typename PCSTy>
class PermutationAssembly {
 public:
  constexpr static size_t kMaxDegree = PCSTy::kMaxDegree;
  constexpr static size_t kRows = kMaxDegree + 1;

  using F = typename PCSTy::Field;
  using Evals = typename PCSTy::Evals;
  using Poly = typename PCSTy::Poly;
  using Domain = typename PCSTy::Domain;
  using Commitment = typename PCSTy::Commitment;
  using Commitments = std::vector<Commitment>;

  PermutationAssembly() = default;

  // Constructor with |PermutationArgument|.
  explicit PermutationAssembly(const PermutationArgument& p)
      : PermutationAssembly(p.columns()) {}

  // Constructor with permutation columns.
  explicit PermutationAssembly(const std::vector<AnyColumnKey>& columns)
      : columns_(columns), cycle_store_(CycleStore(columns_.size(), kRows)) {}

  explicit PermutationAssembly(std::vector<AnyColumnKey>&& columns)
      : columns_(std::move(columns)),
        cycle_store_(CycleStore(columns_.size(), kRows)) {}

  static PermutationAssembly CreateForTesting(std::vector<AnyColumnKey> columns,
                                              CycleStore cycle_store) {
    PermutationAssembly ret;
    ret.columns_ = std::move(columns);
    ret.cycle_store_ = std::move(cycle_store);
    return ret;
  }

  const std::vector<AnyColumnKey>& columns() const { return columns_; }
  const CycleStore& cycle_store() const { return cycle_store_; }

  void Copy(const AnyColumnKey& left_column, size_t left_row,
            const AnyColumnKey& right_column, size_t right_row) {
    CHECK_LE(left_row, kRows);
    CHECK_LE(right_row, kRows);

    // Get indices of each column.
    size_t left_col_idx = GetColumnIndex(left_column);
    size_t right_col_idx = GetColumnIndex(right_column);

    cycle_store_.MergeCycle(Label(left_col_idx, left_row),
                            Label(right_col_idx, right_row));
  }

  // Returns |PermutationVerifyingKey| which has commitments for permutations.
  constexpr PermutationVerifyingKey<PCSTy> BuildVerifyingKey(
      const Entity<PCSTy>* entity,
      const std::vector<Evals>& permutations) const {
    const PCSTy& pcs = entity->pcs();

    Commitments commitments;
    commitments.reserve(columns_.size());
    for (size_t i = 0; i < columns_.size(); ++i) {
      Commitment commitment;
      CHECK(pcs.CommitLagrange(permutations[i], &commitment));
      commitments.push_back(std::move(commitment));
    }

    return PermutationVerifyingKey<PCSTy>(std::move(commitments));
  }

  // Returns the |PermutationProvingKey| that has the coefficient form and
  // evaluation form of the permutation.
  constexpr PermutationProvingKey<Poly, Evals> BuildProvingKey(
      const Prover<PCSTy>* prover,
      const std::vector<Evals>& permutations) const {
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
  std::vector<Evals> GeneratePermutations(const Domain* domain) const {
    UnpermutedTable<Evals> unpermuted_table =
        UnpermutedTable<Evals>::Construct(columns_.size(), domain);

    // Init evaluation formed polynomials with all-zero coefficients.
    std::vector<Evals> permutations =
        base::CreateVector(columns_.size(), Evals::UnsafeZero(kMaxDegree));

    // Assign |unpermuted_table| to |permutations|.
    base::Parallelize(permutations, [&unpermuted_table, this](
                                        absl::Span<Evals> chunk, size_t c,
                                        size_t chunk_size) {
      size_t i = c * chunk_size;
      for (Evals& evals : chunk) {
        for (size_t j = 0; j <= kMaxDegree; ++j) {
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
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ASSEMBLY_H_
