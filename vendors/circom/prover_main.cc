#include "absl/strings/substitute.h"

#include "circomlib/circuit/quadratic_arithmetic_program.h"
#include "circomlib/json/groth16_proof.h"
#include "circomlib/json/json.h"
#include "circomlib/json/prime_field.h"
#include "circomlib/wtns/wtns.h"
#include "circomlib/zkey/zkey.h"
#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/profiler.h"
#include "tachyon/base/time/time.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/bls12_381.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/r1cs/groth16/prove.h"
#include "tachyon/zk/r1cs/groth16/verify.h"

#if TACHYON_CUDA
#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt_holder.h"
#endif

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

struct ProverTime {
  base::TimeDelta total;
  base::TimeDelta max;
  size_t num_runs = 0;

  void Add(base::TimeDelta delta) {
    ++num_runs;
    total += delta;
    max = std::max(delta, max);
  }

  base::TimeDelta GetAvg() const {
    CHECK_NE(num_runs, size_t{0});
    return total / num_runs;
  }
};

struct CreateProofOptions {
  bool no_zk;
  bool no_use_mmap;
  bool verify;
  size_t num_runs;
#if TACHYON_CUDA
  bool disable_fast_twiddles_mode;
#endif
};

template <typename Curve>
void CreateProof(const base::FilePath& zkey_path,
                 const base::FilePath& witness_path,
                 const base::FilePath& proof_path,
                 const base::FilePath& public_path,
                 const CreateProofOptions& options) {
  using F = typename Curve::G1Curve::ScalarField;
  using Domain = math::UnivariateEvaluationDomain<F, SIZE_MAX>;

  Curve::Init();

  base::TimeTicks start = base::TimeTicks::Now();
  std::cout << "Start parsing zkey" << std::endl;
  std::unique_ptr<ZKey<Curve>> zkey =
      ParseZKey<Curve>(zkey_path, !options.no_use_mmap);
  CHECK(zkey);
  zk::r1cs::groth16::ProvingKey<Curve> proving_key =
      zkey->GetProvingKey().ToNativeProvingKey();
  absl::Span<const Coefficient<F>> coefficients = zkey->GetCoefficients();

  base::TimeTicks end = base::TimeTicks::Now();
  std::cout << "Time taken for parsing zkey: " << end - start << std::endl;
  start = end;

  std::cout << "Start parsing witness" << std::endl;
  std::unique_ptr<Wtns<F>> wtns =
      ParseWtns<F>(witness_path, !options.no_use_mmap);
  CHECK(wtns);

  end = base::TimeTicks::Now();
  std::cout << "Time taken for parsing witness: " << end - start << std::endl;
  start = end;

  zk::r1cs::groth16::Proof<Curve> proof;
  absl::Span<const F> full_assignments = wtns->GetWitnesses();
  std::unique_ptr<Domain> domain = Domain::Create(zkey->GetDomainSize());
#if TACHYON_CUDA
  std::cout << "Start initializing Icicle NTT domain" << std::endl;
  math::IcicleNTTHolder<F> icicle_ntt_holder =
      math::IcicleNTTHolder<F>::Create();
  // NOTE(chokobole): For |domain->size()| less than 8, it's very slow to
  // initialize the domain of |IcicleNTT|.
  if (domain->size() >= 8) {
    math::IcicleNTTOptions ntt_options;
    ntt_options.fast_twiddles_mode = !options.disable_fast_twiddles_mode;
    CHECK(icicle_ntt_holder->Init(domain->group_gen(), ntt_options));
    domain->set_icicle(&icicle_ntt_holder);
  }

  end = base::TimeTicks::Now();
  std::cout << "Time taken for initializing Icicle NTT domain: " << end - start
            << std::endl;
  start = end;
#endif
  std::cout << "Start proving" << std::endl;
  ProverTime prover_time;
  size_t num_instance_variables = zkey->GetNumInstanceVariables();
  for (size_t i = 0; i < options.num_runs; ++i) {
    std::vector<F> h_evals =
        QuadraticArithmeticProgram<F>::WitnessMapFromMatrices(
            domain.get(), coefficients, full_assignments);
    if (options.no_zk) {
      proof = zk::r1cs::groth16::CreateProofWithAssignmentNoZK(
          proving_key, absl::MakeConstSpan(h_evals),
          full_assignments.subspan(1, num_instance_variables - 1),
          full_assignments.subspan(num_instance_variables),
          full_assignments.subspan(1));
    } else {
      proof = zk::r1cs::groth16::CreateProofWithAssignmentZK(
          proving_key, absl::MakeConstSpan(h_evals),
          full_assignments.subspan(1, num_instance_variables - 1),
          full_assignments.subspan(num_instance_variables),
          full_assignments.subspan(1));
    }

    end = base::TimeTicks::Now();
    base::TimeDelta delta = end - start;
    std::cout << "Time taken for proving #" << i << ": " << delta << std::endl;
    prover_time.Add(delta);
    start = end;
  }
  std::cout << "Avg time taken for proving: " << prover_time.GetAvg()
            << std::endl;
  std::cout << "Max time taken for proving: " << prover_time.max << std::endl;

  absl::Span<const F> public_inputs =
      full_assignments.subspan(1, num_instance_variables - 1);
  if (options.verify) {
    std::cout << "Start verifying" << std::endl;
    zk::r1cs::groth16::PreparedVerifyingKey<Curve> prepared_verifying_key =
        std::move(proving_key).TakeVerifyingKey().ToPreparedVerifyingKey();
    CHECK(zk::r1cs::groth16::VerifyProof(prepared_verifying_key, proof,
                                         public_inputs))
        << "If you see this error with `--config cuda`, it means your GPU "
           "doesn't have enough RAM for Icicle. Try running it with a GPU with "
           "more RAM.";
    end = base::TimeTicks::Now();
    std::cout << "Time taken for verifying: " << end - start << std::endl;
  }

  CHECK(WriteToJson(proof, proof_path));
  std::cout << "Proof is saved to \"" << proof_path << "\"" << std::endl;
  CHECK(WriteToJson(public_inputs, public_path));
  std::cout << "Public input is saved to \"" << public_path << "\""
            << std::endl;
}

}  // namespace circom

int RealMain(int argc, char** argv) {
  base::FlagParser parser;
  base::FilePath zkey_path;
  base::FilePath witness_path;
  base::FilePath proof_path;
  base::FilePath public_path;
  base::FilePath trace_path;
  Curve curve;
  circom::CreateProofOptions options;
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
  parser.AddFlag<base::FilePathFlag>(&trace_path)
      .set_long_name("--trace_path")
      .set_default_value(base::FilePath("/tmp/circom.perfetto-trace"))
      .set_help("The path to generate trace file");
  parser.AddFlag<base::Flag<Curve>>(&curve)
      .set_long_name("--curve")
      .set_default_value(Curve::kBN254)
      .set_help(
          "The curve type among ('bn254', bls12_381'), by default 'bn254'");
  parser.AddFlag<base::BoolFlag>(&options.no_zk)
      .set_long_name("--no_zk")
      .set_default_value(false)
      .set_help(
          "Create proof without zk. By default zk is enabled. Use this flag to "
          "compare the proof with rapidsnark.");
  parser.AddFlag<base::BoolFlag>(&options.no_use_mmap)
      .set_long_name("--no_use_mmap")
      .set_default_value(false)
      .set_help(
          "Create proof without mmap(2). By default, mmap(2) is enabled, "
          "offering faster proof generation at the cost of increased memory "
          "usage due to the memory mapped file. Use this flag if you want to "
          "use less memory.");
  parser.AddFlag<base::BoolFlag>(&options.verify)
      .set_long_name("--verify")
      .set_default_value(false)
      .set_help(
          "Verify the proof. By default verify is disabled. Use this flag "
          "to verify the proof with the public inputs.");
  parser.AddFlag<base::Flag<size_t>>(&options.num_runs)
      .set_short_name("-n")
      .set_long_name("--num_runs")
      .set_default_value(1)
      .set_help("The number of times to run the proof generation");
#if TACHYON_CUDA
  parser.AddFlag<base::BoolFlag>(&options.disable_fast_twiddles_mode)
      .set_long_name("--disable_fast_twiddles_mode")
      .set_default_value(false)
      .set_help(
          "Disables fast twiddle mode on Icicle NTT domain initialization.");
#endif

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }
#if TACHYON_CUDA
  if (!options.verify) {
    std::cout << "Icicle MSM/NTT may produce incorrect results without any "
                 "error if the polynomial degree is too high relative to the "
                 "GPU RAM size. This may be resolved in the future release of "
                 "the Icicle. In the meantime, please run with --verify to "
                 "ensure the accuracy of the results."
              << std::endl;
  }

#endif
  if (options.num_runs == 0) {
    tachyon_cerr << "num_runs should be positive" << std::endl;
    return 1;
  }

  base::Profiler profiler({trace_path});
  profiler.Init();
  profiler.Start();

  switch (curve) {
    case Curve::kBN254:
      circom::CreateProof<math::bn254::BN254Curve>(
          zkey_path, witness_path, proof_path, public_path, options);
      break;
    case Curve::kBLS12_381:
      circom::CreateProof<math::bls12_381::BLS12_381Curve>(
          zkey_path, witness_path, proof_path, public_path, options);
      break;
  }

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
