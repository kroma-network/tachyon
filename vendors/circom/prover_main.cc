#include "circomlib/circuit/quadratic_arithmetic_program.h"
#include "circomlib/json/groth16_proof.h"
#include "circomlib/json/json.h"
#include "circomlib/json/prime_field.h"
#include "circomlib/wtns/wtns.h"
#include "circomlib/zkey/zkey.h"
#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/bls12_381.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/r1cs/groth16/prove.h"
#include "tachyon/zk/r1cs/groth16/verify.h"
#include "tachyon/base/time/time.h"

namespace tachyon {

enum class Curve {
  kBN254,
  kBLS12_381,
};

namespace base {

template <>
class FlagValueTraits<Curve> {
 public:
  static bool ParseValue(std::string_view input, Curve* value,
                         std::string* reason) {
    if (input == "bn254") {
      *value = Curve::kBN254;
    } else if (input == "bls12_381") {
      *value = Curve::kBLS12_381;
    } else {
      *reason = absl::Substitute("Unknown curve: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base

namespace circom {

template <typename Curve>
void CreateProof(const base::FilePath& zkey_path,
                 const base::FilePath& witness_path,
                 const base::FilePath& proof_path,
                 const base::FilePath& public_path, bool no_zk, bool verify) {
  using F = typename Curve::G1Curve::ScalarField;
  using Domain = math::UnivariateEvaluationDomain<F, SIZE_MAX>;

  Curve::Init();

  zk::r1cs::groth16::ProvingKey<Curve> proving_key;
  zk::r1cs::ConstraintMatrices<F> constraint_matrices;
  {
    std::unique_ptr<ZKey<Curve>> zkey = ParseZKey<Curve>(zkey_path);
    CHECK(zkey);

    proving_key = std::move(*zkey).TakeProvingKey().ToNativeProvingKey();
    constraint_matrices = std::move(*zkey).TakeConstraintMatrices().ToNative();
  }
  std::unique_ptr<Wtns<F>> wtns = ParseWtns<F>(witness_path);
  CHECK(wtns);

  absl::Span<const F> full_assignments = wtns->GetWitnesses();

  std::unique_ptr<Domain> domain =
      Domain::Create(constraint_matrices.num_constraints +
                     constraint_matrices.num_instance_variables);
  std::vector<F> h_evals =
      QuadraticArithmeticProgram<F>::WitnessMapFromMatrices(
          domain.get(), constraint_matrices, full_assignments);

  zk::r1cs::groth16::Proof<Curve> proof;
  if (no_zk) {
    proof = zk::r1cs::groth16::CreateProofWithAssignmentNoZK(
        proving_key, absl::MakeConstSpan(h_evals),
        full_assignments.subspan(
            1, constraint_matrices.num_instance_variables - 1),
        full_assignments.subspan(constraint_matrices.num_instance_variables),
        full_assignments.subspan(1));
  } else {
    proof = zk::r1cs::groth16::CreateProofWithAssignmentZK(
        proving_key, absl::MakeConstSpan(h_evals),
        full_assignments.subspan(
            1, constraint_matrices.num_instance_variables - 1),
        full_assignments.subspan(constraint_matrices.num_instance_variables),
        full_assignments.subspan(1));
  }

  zk::r1cs::groth16::PreparedVerifyingKey<Curve> prepared_verifying_key =
      std::move(proving_key).TakeVerifyingKey().ToPreparedVerifyingKey();
  absl::Span<const F> public_inputs = full_assignments.subspan(
      1, constraint_matrices.num_instance_variables - 1);
  if (verify) {
    CHECK(zk::r1cs::groth16::VerifyProof(prepared_verifying_key, proof,
                                         public_inputs));
  }

  CHECK(WriteToJson(proof, proof_path));
  CHECK(WriteToJson(public_inputs, public_path));
}

}  // namespace circom

int RealMain(int argc, char** argv) {
  base::FlagParser parser;
  base::FilePath zkey_path;
  base::FilePath witness_path;
  base::FilePath proof_path;
  base::FilePath public_path;
  Curve curve;
  bool no_zk = false;
  bool verify = false;
  size_t num_runs;
  parser.AddFlag<base::FilePathFlag>(&zkey_path)
      .set_name("zkey")
      .set_help("The path to zkey file");
  parser.AddFlag<base::FilePathFlag>(&witness_path)
      .set_name("wtns")
      .set_help("The path to wtns file");
  parser.AddFlag<base::FilePathFlag>(&proof_path)
      .set_name("proof")
      .set_help("The path to proof json");
  parser.AddFlag<base::FilePathFlag>(&public_path)
      .set_name("public")
      .set_help("The path to public json");
  parser.AddFlag<base::Flag<Curve>>(&curve).set_long_name("--curve").set_help(
      "The curve type among ('bn254', bls12_381'), by default 'bn254'");
  parser.AddFlag<base::BoolFlag>(&no_zk).set_long_name("--no_zk").set_help(
      "Create proof without zk. By default zk is enabled. Use this flag to "
      "compare the proof with rapidsnark.");
  parser.AddFlag<base::BoolFlag>(&verify)
      .set_long_name("--verify")
      .set_help(
          "Verify the proof. By default verify is disabled. Use this flag "
          "to verify the proof with the public inputs.");
  parser.AddFlag<base::Flag<size_t>>(&num_runs)
      .set_short_name("-n")
      .set_long_name("--num_runs")
      .set_default_value(1)
      .set_required()
      .set_help("The number of times to run the proof generation");

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }
  if (num_runs == 0) {
    tachyon_cerr << "num_runs should be positive" << std::endl;
    return 1;
  }
base::TimeDelta total_time;
base::TimeDelta max_time;
  for (size_t i = 0; i < num_runs; ++i) {
    base::TimeTicks start = base::TimeTicks::Now();
      switch (curve) {
        case Curve::kBN254:
          circom::CreateProof<math::bn254::BN254Curve>(
              zkey_path, witness_path, proof_path, public_path, no_zk, verify);
          break;
        case Curve::kBLS12_381:
          circom::CreateProof<math::bls12_381::BLS12_381Curve>(
              zkey_path, witness_path, proof_path, public_path, no_zk, verify);
          break;
    }
    base::TimeDelta end = base::TimeDelta::Now();
    base::TimeDelta time_taken = end - start;
    total_time += time_taken;
    max_time = std::max(max_time, time_taken);
    std::cout << "Run " << (i + 1) << ", Time Taken: " << time_taken << std::endl;
  }

  base::TimeDelta avg_time = total_time / num_runs;
  std::cout << "Average Time Taken: " << avg_time <<  std::endl;
  std::cout << "Maximum Time Taken: " << max_time <<  std::endl;

  return 0;
}
}  // namespace tachyon
int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
