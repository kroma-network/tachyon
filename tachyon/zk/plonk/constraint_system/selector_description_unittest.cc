#include "tachyon/zk/plonk/constraint_system/selector_description.h"

#include "gtest/gtest.h"

namespace tachyon::zk::plonk {

TEST(SelectorDescriptionTest, IsOrthogonal) {
  std::vector<std::vector<bool>> selector_activations = {
      // [1, 0, 1, 0, 1]
      {true, false, true, false, true},
      // [0, 1, 0, 1, 0]
      {false, true, false, true, false},
      // [1, 0, 0, 0, 0]
      {true, false, false, false, false},
  };
  SelectorDescription a(size_t{0}, &selector_activations[0], size_t{1});
  SelectorDescription b(size_t{1}, &selector_activations[1], size_t{1});
  SelectorDescription c(size_t{2}, &selector_activations[2], size_t{1});

  // [1, 0, 1, 0, 1]
  // [0, 1, 0, 1, 0]
  EXPECT_TRUE(a.IsOrthogonal(b));

  // [0, 1, 0, 1, 0]
  // [1, 0, 0, 0, 0]
  EXPECT_TRUE(b.IsOrthogonal(c));

  // [1, 0, 1, 0, 1]
  // [1, 0, 0, 0, 0]
  // They are not Orthogonal due to the first index.
  EXPECT_FALSE(a.IsOrthogonal(c));
}

}  // namespace tachyon::zk::plonk
