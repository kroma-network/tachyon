// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_argument.h"

#include "gtest/gtest.h"

#include "tachyon/base/random.h"

namespace tachyon::zk {

TEST(PermutationArgumentTest, AddColumn) {
  std::vector<AnyColumn> columns = {
      FixedColumn(0), AdviceColumn(1, kSecondPhase), InstanceColumn(2)};
  PermutationArgument argument(columns);
  EXPECT_EQ(argument.columns(), columns);

  // Already included column should not be added.
  AnyColumn col = base::UniformElement(columns);
  argument.AddColumn(col);
  EXPECT_EQ(argument.columns(), columns);

  // Advice column whose phase is different should be added.
  AdviceColumn col2(1, Phase(3));
  argument.AddColumn(col2);
  columns.push_back(col2);
  EXPECT_EQ(argument.columns(), columns);

  // New Instance column should be added.
  InstanceColumn col3(3);
  argument.AddColumn(col3);
  columns.push_back(col3);
  EXPECT_EQ(argument.columns(), columns);
}

}  // namespace tachyon::zk
