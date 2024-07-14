#include "circomlib/circuit/circuit_test.h"
#include "circomlib/circuit/quadratic_arithmetic_program.h"
#include "circomlib/r1cs/r1cs.h"
#include "circomlib/zkey/zkey.h"
#include "tachyon/zk/r1cs/constraint_system/quadratic_arithmetic_program.h"

namespace tachyon::circom {

class AdderCircuitTest : public CircuitTest {
 public:
  void SetUp() override {
    r1cs_ = ParseR1CS<F>(base::FilePath("examples/adder.r1cs"));
    ASSERT_TRUE(r1cs_);
  }

  void LoadRandomWitness() {
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

  void LoadWitnessFromJson() {
    circuit_.reset(new Circuit<F>(
        r1cs_.get(), base::FilePath("examples/adder_cpp/adder.dat")));
    circuit_->witness_loader().Load(
        base::FilePath("circomlib/circuit/adder_data.json"));
    std::vector<F> public_inputs = circuit_->GetPublicInputs();
    ASSERT_EQ(public_inputs.size(), 1);
    ASSERT_EQ(public_inputs[0], F(7));
  }
};

TEST_F(AdderCircuitTest, Synthesize) {
  LoadRandomWitness();
  this->SynthesizeTest();

  LoadWitnessFromJson();
  this->SynthesizeTest();
}

TEST_F(AdderCircuitTest, Groth16ProveAndVerify) {
  constexpr size_t kMaxDegree = 127;
  LoadRandomWitness();
  this->Groth16ProveAndVerifyTest<kMaxDegree,
                                  zk::r1cs::QuadraticArithmeticProgram<F>>();
}

TEST_F(AdderCircuitTest, Groth16ProveAndVerifyUsingZkey) {
  constexpr size_t kMaxDegree = 127;

  LoadRandomWitness();
  std::unique_ptr<ZKey<Curve>> zkey =
      ParseZKey<Curve>(base::FilePath("examples/adder.zkey"));
  ASSERT_TRUE(zkey);

  std::vector<F> full_assignments = base::CreateVector(
      r1cs_->GetNumVariables(),
      [this](size_t i) { return circuit_->witness_loader().Get(i); });

  this->Groth16ProveAndVerifyUsingZKeyTest<kMaxDegree,
                                           QuadraticArithmeticProgram<F>>(
      *zkey, full_assignments);
}

}  // namespace tachyon::circom
