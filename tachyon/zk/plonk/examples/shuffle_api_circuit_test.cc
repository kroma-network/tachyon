#include "tachyon/zk/plonk/examples/shuffle_api_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/shuffle_api_circuit_test_data.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk {

namespace {

template <typename TestArguments>
class ShuffleAPICircuitTest
    : public CircuitTest<TestArguments, ShuffleAPICircuitTestData<
                                            typename TestArguments::Circuit,
                                            typename TestArguments::PS>> {
 public:
  using Circuit = typename TestArguments::Circuit;
  using PS = typename TestArguments::PS;
  using PCS = typename PS::PCS;
  using Commitment = typename PCS::Commitment;

  static void SetUpTestSuite() {
    CircuitTest<TestArguments,
                ShuffleAPICircuitTestData<Circuit, PS>>::SetUpTestSuite();
    halo2::ProofSerializer<Commitment>::s_use_legacy_serialization = false;
  }
};

}  // namespace

// clang-format off
using ShuffleAPICircuitTestArgumentsList = testing::Types<
    TestArguments<ShuffleAPICircuit<BN254SHPlonk::Field, SimpleFloorPlanner>, BN254SHPlonkLogDerivativeHalo2>,
    TestArguments<ShuffleAPICircuit<BN254SHPlonk::Field, V1FloorPlanner>, BN254SHPlonkLogDerivativeHalo2>>;
// clang-format on

TYPED_TEST_SUITE(ShuffleAPICircuitTest, ShuffleAPICircuitTestArgumentsList);

TYPED_TEST(ShuffleAPICircuitTest, Configure) { this->ConfigureTest(); }
TYPED_TEST(ShuffleAPICircuitTest, Synthesize) { this->SynthesizeTest(); }
TYPED_TEST(ShuffleAPICircuitTest, LoadVerifyingKey) {
  this->LoadVerifyingKeyTest();
}
TYPED_TEST(ShuffleAPICircuitTest, LoadProvingKey) {
  this->LoadProvingKeyTest();
}
TYPED_TEST(ShuffleAPICircuitTest, CreateProof) { this->CreateProofTest(); }
TYPED_TEST(ShuffleAPICircuitTest, VerifyProof) { this->VerifyProofTest(); }

}  // namespace tachyon::zk::plonk
