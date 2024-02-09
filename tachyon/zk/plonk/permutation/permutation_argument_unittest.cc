// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_argument.h"

#include "gtest/gtest.h"

#include "tachyon/base/random.h"
#include "tachyon/zk/plonk/base/ref_table.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"
#include "tachyon/zk/plonk/permutation/permutation_argument_runner.h"
#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

namespace tachyon::zk::plonk {

class PermutationArgumentTest : public halo2::ProverTest {
 public:
  void SetUp() override {
    halo2::ProverTest::SetUp();

    const Domain* domain = prover_->domain();

    Evals cycled_column = domain->Random<Evals>();
    fixed_columns_ = {cycled_column, domain->Random<Evals>(),
                      domain->Random<Evals>()};
    advice_columns_ = {domain->Random<Evals>(), cycled_column,
                       domain->Random<Evals>()};
    instance_columns_ = {cycled_column, domain->Random<Evals>(),
                         domain->Random<Evals>()};

    table_ = RefTable<Evals>(absl::MakeConstSpan(fixed_columns_),
                             absl::MakeConstSpan(advice_columns_),
                             absl::MakeConstSpan(instance_columns_));

    column_keys_ = {
        FixedColumnKey(0), AdviceColumnKey(0), InstanceColumnKey(0),
        FixedColumnKey(1), AdviceColumnKey(1), InstanceColumnKey(1),
        FixedColumnKey(2), AdviceColumnKey(2), InstanceColumnKey(2),
    };
    argument_ = PermutationArgument(column_keys_);

    unpermuted_table_ = UnpermutedTable<Evals>::Construct(
        column_keys_.size(), prover_->pcs().N(), prover_->domain());
  }

 protected:
  std::vector<Evals> fixed_columns_;
  std::vector<Evals> advice_columns_;
  std::vector<Evals> instance_columns_;
  RefTable<Evals> table_;

  std::vector<AnyColumnKey> column_keys_;
  PermutationArgument argument_;
  UnpermutedTable<Evals> unpermuted_table_;
};

TEST_F(PermutationArgumentTest, AddColumn) {
  // Already included column should not be added.
  AnyColumnKey col = base::UniformElement(column_keys_);
  argument_.AddColumn(col);
  EXPECT_EQ(argument_.columns(), column_keys_);

  // Advice column whose phase is different should be added.
  AdviceColumnKey col2(1, Phase(3));
  argument_.AddColumn(col2);
  column_keys_.push_back(col2);
  EXPECT_EQ(argument_.columns(), column_keys_);

  // New Instance column should be added.
  InstanceColumnKey col3(3);
  argument_.AddColumn(col3);
  column_keys_.push_back(col3);
  EXPECT_EQ(argument_.columns(), column_keys_);
}

// TODO(chokobole): Implement test codes correctly. This just test compilation.
TEST_F(PermutationArgumentTest, Commit) {
  prover_->blinder().set_blinding_factors(5);

  size_t n = prover_->pcs().N();
  PermutationAssembly assembly = PermutationAssembly::CreateForTesting(
      column_keys_, CycleStore(column_keys_.size(), n), n);

  std::vector<Evals> permutations =
      assembly.GeneratePermutations<Evals>(prover_->domain());
  PermutationProvingKey<Poly, Evals> pk =
      assembly.BuildProvingKey(prover_.get(), std::move(permutations));

  F beta = F::Random();
  F gamma = F::Random();

  PermutationArgumentRunner<Poly, Evals>::CommitArgument(
      prover_.get(), argument_, table_, n, pk, beta, gamma);
}

}  // namespace tachyon::zk::plonk
