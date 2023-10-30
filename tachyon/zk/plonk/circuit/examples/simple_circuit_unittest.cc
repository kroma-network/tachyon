#include "tachyon/zk/plonk/circuit/examples/simple_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::zk {

namespace {

class SimpleCircuitTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

// TODO(chokobole): Implement test codes.
TEST_F(SimpleCircuitTest, ProveAndVerify) {
  SimpleCircuit<math::bn254::Fr> circuit;
}

}  // namespace tachyon::zk
