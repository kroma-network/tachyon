#include "tachyon/zk/plonk/examples/simple_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/simple_circuit_test_data.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk {

namespace {

template <typename TestArguments>
class SimpleCircuitTest
    : public CircuitTest<TestArguments,
                         SimpleTestData<typename TestArguments::Circuit,
                                        typename TestArguments::PCS,
                                        typename TestArguments::LS>> {};

}  // namespace

using SimpleTestArgumentsList = testing::Types<
    TestArguments<SimpleCircuit<BN254SHPlonk::Field, SimpleFloorPlanner>,
                  BN254SHPlonk, BN254Halo2LS>,
    TestArguments<SimpleCircuit<BN254SHPlonk::Field, V1FloorPlanner>,
                  BN254SHPlonk, BN254Halo2LS>>;

TYPED_TEST_SUITE(SimpleCircuitTest, SimpleTestArgumentsList);

TYPED_TEST(SimpleCircuitTest, Configure) { this->ConfigureTest(); }
TYPED_TEST(SimpleCircuitTest, Synthesize) { this->SynthesizeTest(); }
TYPED_TEST(SimpleCircuitTest, LoadVerifyingKey) {
  this->LoadVerifyingKeyTest();
}
TYPED_TEST(SimpleCircuitTest, LoadProvingKey) { this->LoadProvingKeyTest(); }
TYPED_TEST(SimpleCircuitTest, CreateProof) { this->CreateProofTest(); }
TYPED_TEST(SimpleCircuitTest, VerifyProof) { this->VerifyProofTest(); }

}  // namespace tachyon::zk::plonk
