#include "tachyon/zk/plonk/examples/fibonacci/fibonacci2_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/fibonacci/fibonacci2_circuit_test_data.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk {

namespace {

template <typename TestArguments>
class Fibonacci2CircuitTest
    : public CircuitTest<TestArguments,
                         Fibonacci2TestData<typename TestArguments::Circuit,
                                            typename TestArguments::PCS,
                                            typename TestArguments::LS>> {};

}  // namespace

using Fibonacci2TestArgumentsList = testing::Types<
    TestArguments<Fibonacci2Circuit<BN254SHPlonk::Field, SimpleFloorPlanner>,
                  BN254SHPlonk, BN254Halo2LS>,
    TestArguments<Fibonacci2Circuit<BN254SHPlonk::Field, V1FloorPlanner>,
                  BN254SHPlonk, BN254Halo2LS>>;

TYPED_TEST_SUITE(Fibonacci2CircuitTest, Fibonacci2TestArgumentsList);

TYPED_TEST(Fibonacci2CircuitTest, Configure) { this->ConfigureTest(); }
TYPED_TEST(Fibonacci2CircuitTest, Synthesize) { this->SynthesizeTest(); }
TYPED_TEST(Fibonacci2CircuitTest, LoadVerifyingKey) {
  this->LoadVerifyingKeyTest();
}
TYPED_TEST(Fibonacci2CircuitTest, LoadProvingKey) {
  this->LoadProvingKeyTest();
}
TYPED_TEST(Fibonacci2CircuitTest, CreateProof) { this->CreateProofTest(); }
TYPED_TEST(Fibonacci2CircuitTest, VerifyProof) { this->VerifyProofTest(); }

}  // namespace tachyon::zk::plonk
