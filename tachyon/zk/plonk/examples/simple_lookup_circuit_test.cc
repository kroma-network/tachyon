#include "tachyon/zk/plonk/examples/simple_lookup_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/simple_lookup_circuit_test_data.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk {

namespace {

static const size_t kBits = 3;

template <typename TestArguments>
class SimpleLookupCircuitTest
    : public CircuitTest<TestArguments,
                         SimpleLookupTestData<typename TestArguments::Circuit,
                                              typename TestArguments::PS>> {};

}  // namespace

// clang-format off
using SimpleLookupTestArgumentsList = testing::Types<
    TestArguments<SimpleLookupCircuit<BN254SHPlonk::Field, kBits, SimpleFloorPlanner>, BN254SHPlonkHalo2>,
    TestArguments<SimpleLookupCircuit<BN254SHPlonk::Field, kBits, V1FloorPlanner>, BN254SHPlonkHalo2>>;
// clang-format on

TYPED_TEST_SUITE(SimpleLookupCircuitTest, SimpleLookupTestArgumentsList);

TYPED_TEST(SimpleLookupCircuitTest, Configure) { this->ConfigureTest(); }
TYPED_TEST(SimpleLookupCircuitTest, Synthesize) { this->SynthesizeTest(); }
TYPED_TEST(SimpleLookupCircuitTest, LoadVerifyingKey) {
  this->LoadVerifyingKeyTest();
}
TYPED_TEST(SimpleLookupCircuitTest, LoadProvingKey) {
  this->LoadProvingKeyTest();
}
TYPED_TEST(SimpleLookupCircuitTest, CreateProof) { this->CreateProofTest(); }
TYPED_TEST(SimpleLookupCircuitTest, VerifyProof) { this->VerifyProofTest(); }

}  // namespace tachyon::zk::plonk
