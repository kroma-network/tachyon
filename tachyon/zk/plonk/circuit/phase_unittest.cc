#include "tachyon/zk/plonk/circuit/phase.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::zk {

TEST(PhaseTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(kFirstPhase, kSecondPhase)));
}

}  // namespace tachyon::zk
