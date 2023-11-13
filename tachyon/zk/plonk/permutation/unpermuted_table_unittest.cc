// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/unpermuted_table.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2_prover_test.h"

namespace tachyon::zk {

namespace {

class UnpermutedTableTest : public Halo2ProverTest {
 public:
  constexpr static size_t kCols = 4;

  void SetUp() override {
    Halo2ProverTest::SetUp();
    unpermuted_table_ =
        UnpermutedTable<PCS>::Construct(kCols, prover_->domain());
  }

 protected:
  UnpermutedTable<PCS> unpermuted_table_;
};

}  // namespace

TEST_F(UnpermutedTableTest, Construct) {
  const Domain* domain = prover_->domain();

  const F& omega = domain->group_gen();
  std::vector<F> omega_powers = domain->GetRootsOfUnity(kMaxDegree + 1, omega);

  const F delta = unpermuted_table_.GetDelta();
  EXPECT_NE(delta, F::One());
  EXPECT_EQ(delta.Pow(F::Config::kTrace), F::One());
  for (size_t i = 1; i < kCols; ++i) {
    for (size_t j = 0; j < kMaxDegree + 1; ++j) {
      omega_powers[j] *= delta;
      EXPECT_EQ(omega_powers[j], unpermuted_table_[Label(i, j)]);
    }
  }
}

TEST_F(UnpermutedTableTest, GetColumns) {
  using Col = UnpermutedTable<PCS>::Col;

  const Domain* domain = prover_->domain();

  const F& omega = domain->group_gen();
  const F delta = unpermuted_table_.GetDelta();
  std::vector<F> omega_powers = domain->GetRootsOfUnity(kMaxDegree + 1, omega);
  for (F& omega_power : omega_powers) {
    omega_power *= delta;
  }

  Ref<const Col> column = unpermuted_table_.GetColumn(1);
  for (size_t i = 0; i < kMaxDegree + 1; ++i) {
    EXPECT_EQ(omega_powers[i], column[i]);
  }

  std::vector<Ref<const Col>> columns =
      unpermuted_table_.GetColumns(base::Range<size_t>(2, 4));
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < kMaxDegree + 1; ++j) {
      omega_powers[j] *= delta;
      EXPECT_EQ(omega_powers[j], columns[i][j]);
    }
  }
}

}  // namespace tachyon::zk
