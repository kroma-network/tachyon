#ifndef VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_CIRCUIT_TEST_H_
#define VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_CIRCUIT_TEST_H_

#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "circomlib/circuit/circuit.h"
#include "circomlib/r1cs/r1cs.h"
#include "circomlib/zkey/zkey.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/r1cs/groth16/prove.h"
#include "tachyon/zk/r1cs/groth16/verify.h"

namespace tachyon::circom {

using F = math::bn254::Fr;
using Curve = math::bn254::BN254Curve;

class CircuitTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Curve::Init(); }

 protected:
  void SynthesizeTest() {
    zk::r1cs::ConstraintSystem<F> constraint_system;
    circuit_->Synthesize(constraint_system);
    ASSERT_TRUE(constraint_system.IsSatisfied());
  }

  template <size_t MaxDegree, typename QAP>
  void Groth16ProveAndVerifyTest() {
    zk::r1cs::groth16::ToxicWaste<Curve> toxic_waste =
        zk::r1cs::groth16::ToxicWaste<Curve>::RandomWithoutX();
    zk::r1cs::groth16::ProvingKey<Curve> pk;
    bool loaded = pk.Load<MaxDegree, QAP>(toxic_waste, *circuit_);
    ASSERT_TRUE(loaded);
    zk::r1cs::groth16::Proof<Curve> proof =
        zk::r1cs::groth16::CreateProofWithReductionZK<MaxDegree, QAP>(*circuit_,
                                                                      pk);
    zk::r1cs::groth16::PreparedVerifyingKey<Curve> pvk =
        std::move(pk).TakeVerifyingKey().ToPreparedVerifyingKey();
    std::vector<F> public_inputs = circuit_->GetPublicInputs();
    ASSERT_TRUE(VerifyProof(pvk, proof, public_inputs));
  }

  template <size_t MaxDegree, typename QAP>
  void Groth16ProveAndVerifyUsingZKeyTest(
      const ZKey<Curve>& zkey, absl::Span<const F> full_assignments) {
    using Domain = math::UnivariateEvaluationDomain<F, MaxDegree>;

    zk::r1cs::groth16::ProvingKey<Curve> pk =
        zkey.GetProvingKey().ToNativeProvingKey();
    absl::Span<const Coefficient<F>> coefficients = zkey.GetCoefficients();

    std::unique_ptr<Domain> domain = Domain::Create(zkey.GetDomainSize());
    std::pmr::vector<F> h_evals = QAP::WitnessMapFromMatrices(
        domain.get(), coefficients, full_assignments);

    size_t num_instance_variables = zkey.GetNumInstanceVariables();
    zk::r1cs::groth16::Proof<Curve> proof =
        zk::r1cs::groth16::CreateProofWithAssignmentZK(
            pk, absl::MakeConstSpan(h_evals),
            full_assignments.subspan(1, num_instance_variables - 1),
            full_assignments.subspan(num_instance_variables),
            full_assignments.subspan(1));
    zk::r1cs::groth16::PreparedVerifyingKey<Curve> pvk =
        std::move(pk).TakeVerifyingKey().ToPreparedVerifyingKey();
    std::vector<F> public_inputs = circuit_->GetPublicInputs();
    ASSERT_TRUE(VerifyProof(pvk, proof, public_inputs));
  }

  std::unique_ptr<R1CS<F>> r1cs_;
  std::unique_ptr<Circuit<F>> circuit_;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_CIRCUIT_TEST_H_
