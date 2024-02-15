// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_argument.h"

#include "gtest/gtest.h"

#include "tachyon/base/random.h"

namespace tachyon::zk::plonk {

TEST(PermutationArgumentTest, AddColumn) {
  std::vector<AnyColumnKey> column_keys = {
      FixedColumnKey(0), AdviceColumnKey(0), InstanceColumnKey(0),
      FixedColumnKey(1), AdviceColumnKey(1), InstanceColumnKey(1),
      FixedColumnKey(2), AdviceColumnKey(2), InstanceColumnKey(2),
  };
  PermutationArgument argument(column_keys);

  // Already included column should not be added.
  AnyColumnKey col = base::UniformElement(column_keys);
  argument.AddColumn(col);
  EXPECT_EQ(argument.columns(), column_keys);

  // Advice column whose phase is different should be added.
  AdviceColumnKey col2(1, Phase(3));
  argument.AddColumn(col2);
  column_keys.push_back(col2);
  EXPECT_EQ(argument.columns(), column_keys);

  // New Instance column should be added.
  InstanceColumnKey col3(3);
  argument.AddColumn(col3);
  column_keys.push_back(col3);
  EXPECT_EQ(argument.columns(), column_keys);
}

}  // namespace tachyon::zk::plonk
