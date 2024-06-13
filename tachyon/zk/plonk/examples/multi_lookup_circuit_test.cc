#include "tachyon/zk/plonk/examples/multi_lookup_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/multi_lookup_circuit_test_data.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"

namespace tachyon::zk::plonk {

namespace {

template <typename TestArguments>
class MultiLookupCircuitTest
    : public CircuitTest<TestArguments,
                         MultiLookupTestData<typename TestArguments::Circuit,
                                             typename TestArguments::PCS,
                                             typename TestArguments::LS>> {};

}  // namespace

// clang-format off
using MultiLookupTestArgumentsList = testing::Types<
    TestArguments<MultiLookupCircuit<BN254SHPlonk::Field, SimpleFloorPlanner>, BN254SHPlonk, BN254LogDerivativeHalo2LS>,
    TestArguments<MultiLookupCircuit<BN254GWC::Field, SimpleFloorPlanner>, BN254GWC, BN254LogDerivativeHalo2LS>>;
// clang-format on

TYPED_TEST_SUITE(MultiLookupCircuitTest, MultiLookupTestArgumentsList);

TYPED_TEST(MultiLookupCircuitTest, CreateProof) { this->CreateProofTest(); }
TYPED_TEST(MultiLookupCircuitTest, VerifyProof) { this->VerifyProofTest(); }

}  // namespace tachyon::zk::plonk
