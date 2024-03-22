#include "circomlib/circuit/circuit_test.h"
#include "circomlib/r1cs/r1cs_parser.h"
#include "tachyon/zk/r1cs/constraint_system/quadratic_arithmetic_program.h"

namespace tachyon::circom {

class AdderCircuitTest : public CircuitTest {
 public:
  void SetUp() override {
    R1CSParser parser;
    r1cs_ = parser.Parse(base::FilePath("examples/adder.r1cs"));
    ASSERT_TRUE(r1cs_);

    circuit_.reset(new Circuit<F>(
        r1cs_.get(), base::FilePath("examples/adder_cpp/adder.dat")));
    std::vector<uint32_t> values = base::CreateVector(
        2, []() { return base::Uniform(base::Range<uint32_t>()); });
    circuit_->witness_loader().Set("a", F(values[0]));
    circuit_->witness_loader().Set("b", F(values[1]));
    circuit_->witness_loader().Load();
    std::vector<F> public_inputs = circuit_->GetPublicInputs();
    ASSERT_EQ(public_inputs.size(), 1);
    ASSERT_EQ(public_inputs[0], F(values[0] + values[1]));
  }
};

TEST_F(AdderCircuitTest, Synthesize) { this->SynthesizeTest(); }

TEST_F(AdderCircuitTest, Groth16ProveAndVerify) {
  constexpr size_t kMaxDegree = 127;
  this->Groth16ProveAndVerifyTest<kMaxDegree,
                                  zk::r1cs::QuadraticArithmeticProgram<F>>();
}

}  // namespace tachyon::circom
