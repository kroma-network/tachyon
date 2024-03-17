#include "circomlib/circuit/circuit_test.h"
#include "circomlib/r1cs/r1cs_parser.h"

namespace tachyon::circom {

class Multiplier3CircuitTest : public CircuitTest {
 public:
  void SetUp() override {
    R1CSParser parser;
    r1cs_ = parser.Parse(base::FilePath("examples/multiplier_3.r1cs"));
    ASSERT_TRUE(r1cs_);

    circuit_.reset(new Circuit<F>(
        r1cs_.get(),
        base::FilePath("examples/multiplier_3_cpp/multiplier_3.dat")));
    std::vector<F> values = base::CreateVector(3, []() { return F::Random(); });
    circuit_->witness_loader().Set("in", values);
    circuit_->witness_loader().Load();
    std::vector<F> public_inputs = circuit_->GetPublicInputs();
    ASSERT_EQ(public_inputs.size(), 1);
    ASSERT_EQ(public_inputs[0], values[0] * values[1] * values[2]);
  }
};

TEST_F(Multiplier3CircuitTest, Synthesize) { this->SynthesizeTest(); }

TEST_F(Multiplier3CircuitTest, Groth16ProveAndVerify) {
  constexpr size_t kMaxDegree = 31;
  this->Groth16ProveAndVerifyTest<kMaxDegree>();
}

}  // namespace tachyon::circom
