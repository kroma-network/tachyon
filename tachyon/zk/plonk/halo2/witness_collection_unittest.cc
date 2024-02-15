// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/halo2/witness_collection.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/math/base/rational_field.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/base/value.h"

namespace tachyon::zk::plonk::halo2 {
namespace {

using F = math::bn254::Fr;

class WitnessCollectionTest : public math::FiniteFieldTest<F> {
 public:
  constexpr static RowIndex N = 32;
  constexpr static size_t kMaxDegree = N - 1;
  constexpr static size_t kNumAdviceColumns = 3;
  constexpr static RowIndex kBlindingFactors = 5;
  constexpr static RowIndex kUsableRows = N - (kBlindingFactors + 1);

  using Domain = math::UnivariateEvaluationDomain<F, kMaxDegree>;
  using Evals = Domain::Evals;
  using RationalEvals =
      math::UnivariateEvaluations<math::RationalField<F>, kMaxDegree>;

  void SetUp() override {
    std::unique_ptr<Domain> domain = Domain::Create(N);

    // There is a single challenge in |expected_challenges_|.
    expected_challenges_[0] = F::Random();
    expected_instance_columns_ = {domain->Random<Evals>()};

    Phase current_phase(0);
    witness_collection_ = WitnessCollection<Evals, RationalEvals>(
        domain.get(), kNumAdviceColumns, kUsableRows, current_phase,
        expected_challenges_, expected_instance_columns_);
  }

 protected:
  absl::btree_map<size_t, F> expected_challenges_;
  std::vector<Evals> expected_instance_columns_;
  WitnessCollection<Evals, RationalEvals> witness_collection_;
};

}  // namespace

TEST_F(WitnessCollectionTest, QueryInstance) {
  size_t col = 0;
  RowIndex row = 10;

  // Query a value in specific instance column.
  Value<F> queried_value =
      witness_collection_.QueryInstance(InstanceColumnKey(col), row);
  EXPECT_EQ(queried_value,
            Value<F>::Known(expected_instance_columns_[col][row]));
}

TEST_F(WitnessCollectionTest, AssignAdvice) {
  math::RationalField<F> value_to_be_assign(F::Random());
  size_t col = 0;
  RowIndex row = 10;

  // Assign a random value to specific cell.
  witness_collection_.AssignAdvice(
      "", AdviceColumnKey(col), row, [value_to_be_assign]() {
        return Value<math::RationalField<F>>::Known(value_to_be_assign);
      });

  std::vector<RationalEvals> rational_advice_columns =
      std::move(witness_collection_).TakeAdvices();
  const RationalEvals& rational_column = rational_advice_columns[col];
  EXPECT_EQ(value_to_be_assign, rational_column[row]);
}

TEST_F(WitnessCollectionTest, GetChallenge) {
  // Get challenge value.
  size_t target_idx = 0;
  Value<F> challenge =
      witness_collection_.GetChallenge(Challenge(target_idx, kFirstPhase));
  EXPECT_EQ(Value<F>::Known(expected_challenges_[target_idx]), challenge);
}

}  // namespace tachyon::zk::plonk::halo2
