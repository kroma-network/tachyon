#include "tachyon/zk/plonk/base/phase.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::zk::plonk {

TEST(PhaseTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(kFirstPhase, kSecondPhase)));
}

}  // namespace tachyon::zk::plonk
