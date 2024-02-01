// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/unpermuted_table.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk::plonk {

namespace {

class UnpermutedTableTest : public halo2::ProverTest {
 public:
  constexpr static size_t kCols = 4;

  void SetUp() override {
    halo2::ProverTest::SetUp();
    unpermuted_table_ = UnpermutedTable<Evals>::Construct(
        kCols, prover_->pcs().N(), prover_->domain());
  }

 protected:
  UnpermutedTable<Evals> unpermuted_table_;
};

}  // namespace

TEST_F(UnpermutedTableTest, Construct) {
  const Domain* domain = prover_->domain();

  const F& omega = domain->group_gen();
  RowIndex n = static_cast<RowIndex>(prover_->pcs().N());
  std::vector<F> omega_powers = domain->GetRootsOfUnity(n, omega);

  const F delta = GetDelta<F>();
  for (size_t i = 1; i < kCols; ++i) {
    for (RowIndex j = 0; j < n; ++j) {
      omega_powers[j] *= delta;
      EXPECT_EQ(omega_powers[j], unpermuted_table_[Label(i, j)]);
    }
  }
}

TEST_F(UnpermutedTableTest, GetColumns) {
  const Domain* domain = prover_->domain();

  const F& omega = domain->group_gen();
  const F delta = GetDelta<F>();
  RowIndex n = static_cast<RowIndex>(prover_->pcs().N());
  std::vector<F> omega_powers = domain->GetRootsOfUnity(n, omega);
  for (F& omega_power : omega_powers) {
    omega_power *= delta;
  }

  base::Ref<const Evals> column = unpermuted_table_.GetColumn(1);
  for (RowIndex i = 0; i < n; ++i) {
    EXPECT_EQ(omega_powers[i], *(*column)[i]);
  }

  std::vector<base::Ref<const Evals>> columns =
      unpermuted_table_.GetColumns(base::Range<size_t>(2, 4));
  for (size_t i = 0; i < 2; ++i) {
    for (RowIndex j = 0; j < n; ++j) {
      omega_powers[j] *= delta;
      EXPECT_EQ(omega_powers[j], *(*columns[i])[j]);
    }
  }
}

}  // namespace tachyon::zk::plonk
