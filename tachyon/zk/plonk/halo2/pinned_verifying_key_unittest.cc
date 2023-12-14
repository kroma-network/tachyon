#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk::halo2 {

namespace {

class PinnedVerifyingKeyTest : public ProverTest {};

}  // namespace

TEST_F(PinnedVerifyingKeyTest, ToRustDebugString) {
  // TODO(chokobole): Enable this test once every feature is implemented.
  // This just tests whether compilation is working or not.
  VerifyingKey<PCS> verifying_key;
  PinnedVerifyingKey<PCS> pinned_verifying_key(prover_.get(), verifying_key);
  base::ToRustDebugString(pinned_verifying_key);
}

}  // namespace tachyon::zk::halo2
