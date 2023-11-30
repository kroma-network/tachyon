#include "tachyon/zk/plonk/permutation/permutation_table_store.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2_prover_test.h"

namespace tachyon::zk {
namespace {

class PermutationTableStoreTest : public Halo2ProverTest {
 public:
  static constexpr size_t kChunkSize = 4;

  void SetUp() override {
    Halo2ProverTest::SetUp();

    fixed_columns_ = {Evals::Random(), Evals::Random(), Evals::Random()};
    advice_columns_ = {Evals::Random(), Evals::Random(), Evals::Random()};
    instance_columns_ = {Evals::Random(), Evals::Random(), Evals::Random()};

    table_ = Table<Evals>(absl::MakeConstSpan(fixed_columns_),
                          absl::MakeConstSpan(advice_columns_),
                          absl::MakeConstSpan(instance_columns_));

    column_keys_ = {
        FixedColumnKey(0),  InstanceColumnKey(0), AdviceColumnKey(0),
        AdviceColumnKey(1), FixedColumnKey(1),    FixedColumnKey(2),
        AdviceColumnKey(2), InstanceColumnKey(1), InstanceColumnKey(2)};

    unpermuted_table_ = UnpermutedTable<Evals>::Construct(column_keys_.size(),
                                                          prover_->domain());
    for (const Evals& column : unpermuted_table_.table()) {
      permutations_.push_back(column);
    }
    permuted_table_ = PermutedTable<Evals>(&permutations_);
  }

 protected:
  std::vector<Evals> fixed_columns_;
  std::vector<Evals> advice_columns_;
  std::vector<Evals> instance_columns_;
  Table<Evals> table_;

  std::vector<AnyColumnKey> column_keys_;
  std::vector<Evals> permutations_;
  PermutedTable<Evals> permuted_table_;
  UnpermutedTable<Evals> unpermuted_table_;
};

}  // namespace

TEST_F(PermutationTableStoreTest, GetColumns) {
  PermutationTableStore<Evals> permutation_table_store(
      column_keys_, table_, permuted_table_, unpermuted_table_, kChunkSize);

  // Prepare columns sorted in the order of |column_keys_|
  std::vector<Evals> expected_value_columns = {
      fixed_columns_[0],  instance_columns_[0], advice_columns_[0],
      advice_columns_[1], fixed_columns_[1],    fixed_columns_[2],
      advice_columns_[2], instance_columns_[1], instance_columns_[2],
  };

  for (size_t i = 0; i < permutation_table_store.GetChunkNum(); ++i) {
    std::vector<Ref<const Evals>> value_columns =
        permutation_table_store.GetValueColumns(i);
    std::vector<Ref<const Evals>> permuted_columns =
        permutation_table_store.GetPermutedColumns(i);
    std::vector<Ref<const Evals>> unpermuted_columns =
        permutation_table_store.GetUnpermutedColumns(i);

    size_t start = permutation_table_store.GetChunkOffset(i);
    size_t chunk_size = permutation_table_store.GetChunkSize(i);
    for (size_t j = 0; j < chunk_size; ++j) {
      EXPECT_EQ(*permuted_columns[j], *unpermuted_columns[j]);
      EXPECT_EQ(*value_columns[j], expected_value_columns[start + j]);
    }
  }
}

}  // namespace tachyon::zk
