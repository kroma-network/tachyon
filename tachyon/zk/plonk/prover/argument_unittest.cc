// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/prover/argument.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/base/halo2/halo2_prover_test.h"

namespace tachyon::zk {
namespace {

class ArgumentTest : public Halo2ProverTest {
 public:
  void InitColumns() {
    const Domain* domain = prover_->domain();
    num_circuits_ = 2;
    expected_fixed_columns_ =
        base::CreateVector(1, [domain]() { return domain->Random<Evals>(); });
    expected_fixed_polys_ =
        base::CreateVector(1, [domain]() { return domain->Random<Poly>(); });
    expected_advice_columns_vec_ =
        base::CreateVector(num_circuits_, [domain]() {
          return base::CreateVector(
              2, [domain]() { return domain->Random<Evals>(); });
        });
    expected_advice_blinds_vec_ = base::CreateVector(num_circuits_, []() {
      return base::CreateVector(2, []() { return F::Random(); });
    });
    expected_instance_columns_vec_ =
        base::CreateVector(num_circuits_, [domain]() {
          return base::CreateVector(
              1, [domain]() { return domain->Random<Evals>(); });
        });
    expected_challenges_ = {F::Random()};
  }

  void SetUp() override {
    Halo2ProverTest::SetUp();
    InitColumns();

    // Copy data to be moved.
    std::vector<std::vector<Evals>> advice_columns_vec =
        expected_advice_columns_vec_;
    std::vector<std::vector<F>> advice_blinds_vec = expected_advice_blinds_vec_;
    std::vector<std::vector<Evals>> instance_columns_vec =
        expected_instance_columns_vec_;
    std::vector<F> challenges = expected_challenges_;

    argument_ = Argument<PCS>::Create(
        prover_.get(), num_circuits_, &expected_fixed_columns_,
        &expected_fixed_polys_, std::move(advice_columns_vec),
        std::move(advice_blinds_vec), std::move(instance_columns_vec),
        std::move(challenges));
  }

 protected:
  size_t num_circuits_ = 2;
  std::vector<Evals> expected_fixed_columns_;
  std::vector<Poly> expected_fixed_polys_;
  std::vector<std::vector<Evals>> expected_advice_columns_vec_;
  std::vector<std::vector<F>> expected_advice_blinds_vec_;
  std::vector<std::vector<Evals>> expected_instance_columns_vec_;
  std::vector<F> expected_challenges_;

  Argument<PCS> argument_;
};

}  // namespace

TEST_F(ArgumentTest, ExportEvalsTables) {
  EXPECT_FALSE(argument_.advice_transformed());
  std::vector<Table<Evals>> column_tables = argument_.ExportColumnTables();

  for (size_t i = 0; i < num_circuits_; ++i) {
    absl::Span<const Evals> fixed_columns = column_tables[i].fixed_columns();
    absl::Span<const Evals> advice_columns = column_tables[i].advice_columns();
    absl::Span<const Evals> instance_columns =
        column_tables[i].instance_columns();

    CHECK_EQ(expected_fixed_columns_[0], fixed_columns[0]);
    CHECK_EQ(expected_advice_columns_vec_[i][0], advice_columns[0]);
    CHECK_EQ(expected_advice_columns_vec_[i][1], advice_columns[1]);
    CHECK_EQ(expected_instance_columns_vec_[i][0], instance_columns[0]);
  }
}

TEST_F(ArgumentTest, ExportPolyTables) {
  argument_.TransformAdvice(prover_->domain());
  EXPECT_TRUE(argument_.advice_transformed());
  std::vector<Table<Poly>> poly_tables = argument_.ExportPolyTables();

  for (size_t i = 0; i < num_circuits_; ++i) {
    absl::Span<const Poly> fixed_polys = poly_tables[i].fixed_columns();
    absl::Span<const Poly> advice_polys = poly_tables[i].advice_columns();
    absl::Span<const Poly> instance_polys = poly_tables[i].instance_columns();

    CHECK_EQ(expected_fixed_polys_[0], fixed_polys[0]);
    CHECK_EQ(expected_advice_columns_vec_[i][0],
             prover_->domain()->FFT(advice_polys[0]));
    CHECK_EQ(expected_advice_columns_vec_[i][1],
             prover_->domain()->FFT(advice_polys[1]));
    CHECK_EQ(expected_instance_columns_vec_[i][0],
             prover_->domain()->FFT(instance_polys[0]));
  }
}

TEST_F(ArgumentTest, GetAdviceBlinds) {
  for (size_t i = 0; i < num_circuits_; ++i) {
    std::vector<F> blinds = argument_.GetAdviceBlinds(i);
    EXPECT_EQ(blinds, expected_advice_blinds_vec_[i]);
  }
}

}  // namespace tachyon::zk
