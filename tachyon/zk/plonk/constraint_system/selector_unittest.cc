#include "tachyon/zk/plonk/constraint_system/selector.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::zk::plonk {

TEST(SelectorTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(Selector::Simple(0), Selector::Simple(1),
                      Selector::Complex(0), Selector::Complex(1))));
}

}  // namespace tachyon::zk::plonk
