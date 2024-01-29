// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/cycle_store.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/random.h"

namespace tachyon::zk {

TEST(CycleStoreTest, MergeCycle) {
  constexpr size_t kCols = 10;
  constexpr RowIndex kRows = 10;

  CycleStore store(kCols, kRows);
  std::vector<Label> labels;
  labels.reserve(kCols * kRows);
  for (size_t col = 0; col < kCols; ++col) {
    for (RowIndex row = 0; row < kRows; ++row) {
      Label l(col, row);
      EXPECT_EQ(store.GetCycleBase(l), l);
      EXPECT_EQ(store.GetCycleSize(l), size_t{1});
      labels.push_back(l);
    }
  }

  for (int i = 0; i < 10; ++i) {
    Label label = base::UniformElement(labels);
    Label label2 = base::UniformElement(labels);

    bool belong_to_same_cycle = store.CheckSameCycle(label, label2);
    size_t cycle_size = store.GetCycleSize(label);
    size_t cycle_size2 = store.GetCycleSize(label2);
    EXPECT_NE(belong_to_same_cycle, store.MergeCycle(label, label2));
    if (belong_to_same_cycle) {
      EXPECT_EQ(cycle_size, cycle_size2);
      EXPECT_EQ(cycle_size, store.GetCycleSize(label));
      EXPECT_EQ(cycle_size2, store.GetCycleSize(label2));
    } else {
      EXPECT_EQ(cycle_size + cycle_size2, store.GetCycleSize(label));
      EXPECT_EQ(cycle_size + cycle_size2, store.GetCycleSize(label2));
      EXPECT_TRUE(store.CheckSameCycle(label, label2));
    }
    EXPECT_THAT(store.GetAllLabels(label),
                testing::UnorderedElementsAreArray(store.GetAllLabels(label2)));
  }
}

}  // namespace tachyon::zk
