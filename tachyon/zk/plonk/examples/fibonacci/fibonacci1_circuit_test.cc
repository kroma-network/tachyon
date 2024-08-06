#include "tachyon/zk/plonk/examples/fibonacci/fibonacci1_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/fibonacci/fibonacci1_circuit_test_data.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk {

namespace {

template <typename TestArguments>
class Fibonacci1CircuitTest
    : public CircuitTest<TestArguments,
                         Fibonacci1TestData<typename TestArguments::Circuit,
                                            typename TestArguments::PS>> {};

}  // namespace

using Fibonacci1TestArgumentsList = testing::Types<
    TestArguments<Fibonacci1Circuit<BN254SHPlonk::Field, SimpleFloorPlanner>,
                  BN254SHPlonkHalo2>,
    TestArguments<Fibonacci1Circuit<BN254SHPlonk::Field, V1FloorPlanner>,
                  BN254SHPlonkHalo2>>;

TYPED_TEST_SUITE(Fibonacci1CircuitTest, Fibonacci1TestArgumentsList);

TYPED_TEST(Fibonacci1CircuitTest, Configure) { this->ConfigureTest(); }
TYPED_TEST(Fibonacci1CircuitTest, Synthesize) { this->SynthesizeTest(); }
TYPED_TEST(Fibonacci1CircuitTest, LoadVerifyingKey) {
  this->LoadVerifyingKeyTest();
}
TYPED_TEST(Fibonacci1CircuitTest, LoadProvingKey) {
  this->LoadProvingKeyTest();
}
TYPED_TEST(Fibonacci1CircuitTest, CreateProof) { this->CreateProofTest(); }
TYPED_TEST(Fibonacci1CircuitTest, VerifyProof) { this->VerifyProofTest(); }

}  // namespace tachyon::zk::plonk
