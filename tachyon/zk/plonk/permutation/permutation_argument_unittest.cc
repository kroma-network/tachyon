// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_argument.h"

#include "gtest/gtest.h"

#include "tachyon/base/random.h"
#include "tachyon/zk/base/halo2/halo2_prover_test.h"
#include "tachyon/zk/plonk/circuit/table.h"
#include "tachyon/zk/plonk/permutation/permutation_argument_runner.h"
#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

namespace tachyon::zk {

class PermutationArgumentTest : public Halo2ProverTest {
 public:
  void SetUp() override {
    Halo2ProverTest::SetUp();

    Evals cycled_column = Evals::Random();
    fixed_columns_ = {cycled_column, Evals::Random(), Evals::Random()};
    advice_columns_ = {Evals::Random(), cycled_column, Evals::Random()};
    instance_columns_ = {cycled_column, Evals::Random(), Evals::Random()};

    table_ = Table<Evals>(absl::MakeConstSpan(fixed_columns_),
                          absl::MakeConstSpan(advice_columns_),
                          absl::MakeConstSpan(instance_columns_));

    column_keys_ = {
        FixedColumnKey(0), AdviceColumnKey(0), InstanceColumnKey(0),
        FixedColumnKey(1), AdviceColumnKey(1), InstanceColumnKey(1),
        FixedColumnKey(2), AdviceColumnKey(2), InstanceColumnKey(2),
    };
    argument_ = PermutationArgument(column_keys_);

    unpermuted_table_ = UnpermutedTable<Evals>::Construct(column_keys_.size(),
                                                          prover_->domain());
  }

 protected:
  std::vector<Evals> fixed_columns_;
  std::vector<Evals> advice_columns_;
  std::vector<Evals> instance_columns_;
  Table<Evals> table_;

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
  PermutationAssembly<PCS> assembly =
      PermutationAssembly<PCS>::CreateForTesting(
          column_keys_, CycleStore(column_keys_.size(), kDomainSize),
          kDomainSize);

  std::vector<Evals> permutations =
      assembly.GeneratePermutations(prover_->domain());
  PermutationProvingKey<Poly, Evals> pk =
      assembly.BuildProvingKey(prover_.get(), permutations);

  F beta = F::Random();
  F gamma = F::Random();

  PermutationArgumentRunner<Poly, Evals>::CommitArgument(
      prover_.get(), argument_, table_, prover_->pcs().N(), pk, beta, gamma);
}

}  // namespace tachyon::zk
