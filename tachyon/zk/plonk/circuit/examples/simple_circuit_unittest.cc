#include "tachyon/zk/plonk/circuit/examples/simple_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/zk/base/halo2/halo2_prover_test.h"
#include "tachyon/zk/plonk/keys/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

namespace tachyon::zk {

namespace {

class SimpleCircuitTest : public Halo2ProverTest {};

}  // namespace

// TODO(chokobole): Implement test codes.
TEST_F(SimpleCircuitTest, ProveAndVerify) {
  SimpleCircuit<math::bn254::Fr> circuit;
  VerifyingKey<PCS> vkey;
  ASSERT_TRUE(VerifyingKey<PCS>::Generate(prover_.get(), circuit, &vkey));
}

}  // namespace tachyon::zk
