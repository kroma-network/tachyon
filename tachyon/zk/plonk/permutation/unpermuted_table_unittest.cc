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

class UnpermutedTableTest : public Halo2ProverTest {};

}  // namespace

TEST_F(UnpermutedTableTest, Construct) {
  constexpr size_t kCols = 4;
  const Domain* domain = prover_->domain();

  UnpermutedTable<PCS> unpermuted_table =
      UnpermutedTable<PCS>::Construct(kCols, domain);
  const F& omega = domain->group_gen();
  std::vector<F> omega_powers = domain->GetRootsOfUnity(kMaxDegree + 1, omega);

  const F delta = unpermuted_table.GetDelta();
  EXPECT_NE(delta, F::One());
  EXPECT_EQ(delta.Pow(F::Config::kTrace), F::One());
  for (size_t i = 1; i < kCols; ++i) {
    for (size_t j = 0; j < kMaxDegree + 1; ++j) {
      omega_powers[j] *= delta;
      EXPECT_EQ(omega_powers[j], unpermuted_table[Label(i, j)]);
    }
  }
}
}  // namespace tachyon::zk
