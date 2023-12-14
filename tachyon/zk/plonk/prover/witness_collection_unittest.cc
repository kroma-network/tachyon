// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/prover/witness_collection.h"

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2/halo2_prover_test.h"
#include "tachyon/zk/base/value.h"
#include "tachyon/zk/plonk/circuit/examples/simple_circuit.h"

namespace tachyon::zk {
namespace {

class WitnessCollectionTest : public Halo2ProverTest {
 public:
  using F = typename PCS::Field;
  using RationalEvals = typename PCS::RationalEvals;

  void SetUp() override {
    Halo2ProverTest::SetUp();

    const Domain* domain = prover_->domain();
    prover_->blinder().set_blinding_factors(5);

    // There is a single challenge in |challenges_|.
    expected_challenges_[0] = F::Random();
    expected_instance_columns_ = {domain->Random<Evals>()};

    Phase current_phase(0);
    witness_collection_ = WitnessCollection<PCS>(
        domain, 3, prover_->GetUsableRows(), current_phase,
        expected_challenges_, expected_instance_columns_);
  }

 protected:
  absl::btree_map<size_t, F> expected_challenges_;
  std::vector<Evals> expected_instance_columns_;
  WitnessCollection<PCS> witness_collection_;
};

}  // namespace

TEST_F(WitnessCollectionTest, QueryInstance) {
  size_t col = 0;
  size_t row = 10;

  // Query a value in specific instance column.
  Value<F> queried_value =
      witness_collection_.QueryInstance(InstanceColumnKey(col), row);
  EXPECT_EQ(queried_value,
            Value<F>::Known(*expected_instance_columns_[col][row]));
}

TEST_F(WitnessCollectionTest, AssignAdvice) {
  math::RationalField<F> value_to_be_assign(F::Random());
  size_t col = 0;
  size_t row = 10;

  // Assign a random value to specific cell.
  witness_collection_.AssignAdvice(
      "", AdviceColumnKey(col), row, [value_to_be_assign]() {
        return Value<math::RationalField<F>>::Known(value_to_be_assign);
      });

  std::vector<RationalEvals> rational_advice_columns =
      std::move(witness_collection_).TakeAdvices();
  const RationalEvals& rational_column = rational_advice_columns[col];
  EXPECT_EQ(value_to_be_assign, *rational_column[row]);
}

TEST_F(WitnessCollectionTest, GetChallenge) {
  // Get challenge value.
  size_t target_idx = 0;
  Value<F> challenge =
      witness_collection_.GetChallenge(Challenge(target_idx, kFirstPhase));
  EXPECT_EQ(Value<F>::Known(expected_challenges_[target_idx]), challenge);
}

}  // namespace tachyon::zk
