#include "tachyon/zk/plonk/examples/fibonacci/fibonacci3_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/fibonacci/fibonacci3_circuit_test_data.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk {

namespace {

template <typename TestArguments>
class Fibonacci3CircuitTest
    : public CircuitTest<TestArguments,
                         Fibonacci3TestData<typename TestArguments::Circuit,
                                            typename TestArguments::PS>> {};

}  // namespace

using Fibonacci3TestArgumentsList = testing::Types<
    TestArguments<Fibonacci3Circuit<BN254SHPlonk::Field, SimpleFloorPlanner>,
                  BN254SHPlonkHalo2>,
    TestArguments<Fibonacci3Circuit<BN254SHPlonk::Field, V1FloorPlanner>,
                  BN254SHPlonkHalo2>>;

TYPED_TEST_SUITE(Fibonacci3CircuitTest, Fibonacci3TestArgumentsList);

TYPED_TEST(Fibonacci3CircuitTest, Configure) { this->ConfigureTest(); }
TYPED_TEST(Fibonacci3CircuitTest, Synthesize) { this->SynthesizeTest(); }
TYPED_TEST(Fibonacci3CircuitTest, LoadVerifyingKey) {
  this->LoadVerifyingKeyTest();
}
TYPED_TEST(Fibonacci3CircuitTest, LoadProvingKey) {
  this->LoadProvingKeyTest();
}
TYPED_TEST(Fibonacci3CircuitTest, CreateProof) { this->CreateProofTest(); }
TYPED_TEST(Fibonacci3CircuitTest, VerifyProof) { this->VerifyProofTest(); }

}  // namespace tachyon::zk::plonk
