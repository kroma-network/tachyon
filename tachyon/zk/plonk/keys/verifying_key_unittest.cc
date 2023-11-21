#include "tachyon/zk/plonk/keys/verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2_prover_test.h"
#include "tachyon/zk/plonk/keys/halo2/pinned_verifying_key.h"

namespace tachyon::zk {

namespace {

class VerifyingKeyTest : public Halo2ProverTest {};

}  // namespace

// TODO(chokobole): Check verifying key hash against halo2 one.
TEST_F(VerifyingKeyTest, SetTranscriptRepresentative) {
  VerifyingKey<PCS> verifying_key;
  EXPECT_EQ(verifying_key.transcript_repr(), F::Zero());

  verifying_key.SetTranscriptRepresentative();
  EXPECT_NE(verifying_key.transcript_repr(), F::Zero());
}

// TODO(chokobole): Implement test codes.
TEST_F(VerifyingKeyTest, Generate) { VerifyingKey<PCS> verifying_key; }

}  // namespace tachyon::zk
