#include "circomlib/circuit/circuit_test.h"
#include "circomlib/circuit/quadratic_arithmetic_program.h"
#include "circomlib/r1cs/r1cs_parser.h"
#include "circomlib/zkey/zkey_parser.h"
#include "tachyon/zk/r1cs/constraint_system/quadratic_arithmetic_program.h"

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
  this->Groth16ProveAndVerifyTest<kMaxDegree,
                                  zk::r1cs::QuadraticArithmeticProgram<F>>();
}

TEST_F(Multiplier3CircuitTest, Groth16ProveAndVerifyUsingZkey) {
  constexpr size_t kMaxDegree = 3;

  ZKeyParser parser;
  std::unique_ptr<ZKey> zkey =
      parser.Parse(base::FilePath("examples/multiplier_3.zkey"));
  ASSERT_TRUE(zkey);

  std::vector<F> full_assignments = base::CreateVector(
      r1cs_->GetNumVariables(),
      [this](size_t i) { return circuit_->witness_loader().Get(i); });

  this->Groth16ProveAndVerifyUsingZKeyTest<kMaxDegree,
                                           QuadraticArithmeticProgram<F>>(
      std::move(*zkey), full_assignments);
}

}  // namespace tachyon::circom
