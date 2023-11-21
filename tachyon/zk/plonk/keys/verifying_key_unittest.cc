#include "tachyon/zk/plonk/keys/verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2_prover_test.h"

namespace tachyon::zk {

namespace {

class VerifyingKeyTest : public Halo2ProverTest {};

}  // namespace

// TODO(chokobole): Implement test codes.
TEST_F(VerifyingKeyTest, Generate) { VerifyingKey<PCS> verifying_key; }

}  // namespace tachyon::zk
