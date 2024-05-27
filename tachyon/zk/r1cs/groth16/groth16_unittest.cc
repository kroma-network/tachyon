#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/r1cs/constraint_system/quadratic_arithmetic_program.h"
#include "tachyon/zk/r1cs/constraint_system/test/simple_circuit.h"
#include "tachyon/zk/r1cs/groth16/owned_proving_key.h"
#include "tachyon/zk/r1cs/groth16/prove.h"
#include "tachyon/zk/r1cs/groth16/verify.h"

namespace tachyon::zk::r1cs::groth16 {

namespace {

using F = math::bn254::Fr;
using Curve = math::bn254::BN254Curve;

constexpr size_t MaxDegree = 31;

class Groth16Test : public testing::Test {
 public:
  static void SetUpTestSuite() { Curve::Init(); }
};

}  // namespace

TEST_F(Groth16Test, ProveAndVerify) {
  SimpleCircuit<F> circuit(F::Random(), F::Random());
  ToxicWaste<Curve> toxic_waste = ToxicWaste<Curve>::RandomWithoutX();
  OwnedProvingKey<Curve> opk;
  bool loaded =
      opk.Load<MaxDegree, QuadraticArithmeticProgram<F>>(toxic_waste, circuit);
  ASSERT_TRUE(loaded);
  Proof<Curve> proof =
      CreateProofWithReductionZK<MaxDegree, QuadraticArithmeticProgram<F>>(
          circuit, opk);
  PreparedVerifyingKey<Curve> pvk =
      std::move(opk).TakeOwnedVerifyingKey().ToPreparedVerifyingKey();
  std::vector<F> public_inputs = circuit.GetPublicInputs();
  ASSERT_TRUE(VerifyProof(pvk, proof, public_inputs));

  proof = ReRandomizeProof(pvk.verifying_key(), proof);
  ASSERT_TRUE(VerifyProof(pvk, proof, public_inputs));
}

}  // namespace tachyon::zk::r1cs::groth16
