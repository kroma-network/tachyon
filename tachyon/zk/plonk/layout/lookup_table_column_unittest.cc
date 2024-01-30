#include "tachyon/zk/plonk/layout/lookup_table_column.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::zk {

TEST(LookupTableColumnTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(LookupTableColumn(), LookupTableColumn(FixedColumnKey(1)),
                      LookupTableColumn(FixedColumnKey(2)))));
}

}  // namespace tachyon::zk
