
#include "tachyon/zk/plonk/examples/shuffle_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/shuffle_circuit_test_data.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk {

namespace {

static const size_t kW = 2;
static const size_t kH = 8;

template <typename TestArguments>
class ShuffleCircuitTest
    : public CircuitTest<TestArguments,
                         ShuffleTestData<typename TestArguments::Circuit,
                                         typename TestArguments::PCS,
                                         typename TestArguments::LS>> {};

}  // namespace

// clang-format off
using ShuffleTestArgumentsList = testing::Types<
    TestArguments<ShuffleCircuit<BN254SHPlonk::Field, kW, kH, SimpleFloorPlanner>, BN254SHPlonk, BN254Halo2LS>,
    TestArguments<ShuffleCircuit<BN254SHPlonk::Field, kW, kH, V1FloorPlanner>, BN254SHPlonk, BN254Halo2LS>>;
// clang-format on

TYPED_TEST_SUITE(ShuffleCircuitTest, ShuffleTestArgumentsList);

TYPED_TEST(ShuffleCircuitTest, Configure) { this->ConfigureTest(); }
TYPED_TEST(ShuffleCircuitTest, Synthesize) { this->SynthesizeTest(); }
TYPED_TEST(ShuffleCircuitTest, LoadVerifyingKey) {
  this->LoadVerifyingKeyTest();
}
TYPED_TEST(ShuffleCircuitTest, LoadProvingKey) { this->LoadProvingKeyTest(); }
TYPED_TEST(ShuffleCircuitTest, CreateProof) { this->CreateProofTest(); }
TYPED_TEST(ShuffleCircuitTest, VerifyProof) { this->VerifyProofTest(); }

}  // namespace tachyon::zk::plonk
