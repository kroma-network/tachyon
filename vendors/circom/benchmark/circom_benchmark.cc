#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <utility>

#include "alt_bn128.hpp"  // NOLINT(build/include_subdir)

// clang-format off
#include "benchmark/rapidsnark_runner.h"
#include "benchmark/tachyon_runner.h"
// clang-format on
#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/json/json.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"

namespace tachyon::circom {

using namespace math;

using F = bn254::Fr;
using Curve = math::bn254::BN254Curve;

constexpr size_t MaxDegree = (size_t{1} << 25) - 1;

int RealMain(int argc, char** argv) {
  base::FlagParser parser;
  size_t n;
  parser.AddFlag<base::Flag<size_t>>(&n)
      .set_short_name("-n")
      .set_required()
      .set_help("The number of test to run");
  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }
  if (n == 0) {
    tachyon_cerr << "n should be positive" << std::endl;
    return 1;
  }

  Curve::Init();

  std::vector<std::unique_ptr<Runner<Curve>>> runners;
  runners.push_back(std::make_unique<TachyonRunner<Curve, MaxDegree>>(
      base::FilePath("examples/plonky2/plonky2_cpp/plonky2.dat")));
  runners.push_back(std::make_unique<RapidsnarkRunner<Curve, AltBn128::Engine>>(
      base::FilePath("benchmark/plonky2_verification_key.json")));
  std::vector<zk::r1cs::groth16::Proof<Curve>> proofs;

  std::vector<uint8_t> in = base::CreateVector(
      64, [](size_t i) { return base::Uniform(base::Range<uint8_t>()); });
  std::vector<F> full_assignments;
  absl::Span<const F> public_inputs;

  for (size_t i = 0; i < runners.size(); ++i) {
    std::unique_ptr<Runner<Curve>>& runner = runners[i];
    runner->LoadZkey(base::FilePath("benchmark/plonky2_final.zkey"));

    if (i == 0) {
      TachyonRunner<Curve, MaxDegree>* tachyon_runner =
          reinterpret_cast<TachyonRunner<Curve, MaxDegree>*>(runner.get());

      WitnessLoader<F>& witness_loader = tachyon_runner->witness_loader();
      witness_loader.Load(base::FilePath("benchmark/plonky2_proof.json"));

      const zk::r1cs::ConstraintMatrices<F>& constraint_matrices =
          tachyon_runner->constraint_matrices();

      full_assignments = base::CreateVector(
          constraint_matrices.num_instance_variables +
              constraint_matrices.num_witness_variables,
          [&witness_loader](size_t i) { return witness_loader.Get(i); });

      public_inputs =
          absl::MakeConstSpan(full_assignments)
              .subspan(1, constraint_matrices.num_instance_variables - 1);
    }

    base::TimeDelta total_delta;
    for (size_t j = 0; j < n; ++j) {
      base::TimeDelta delta;
      zk::r1cs::groth16::Proof<Curve> proof =
          runner->Run(full_assignments, public_inputs, delta);
      if (j == 0) {
        proofs.push_back(proof);
      }
      std::cout << "[" << j << "]: " << delta << std::endl;
      total_delta += delta;
    }
    if (i == 0) {
      std::cout << "tachyon(avg): ";
    } else {
      std::cout << "rapidsnark(avg): ";
    }
    std::cout << total_delta / n << std::endl;

    if (i > 0) {
      CHECK_EQ(proofs[0], proofs.back());
    }
  }
  return 0;
}

}  // namespace tachyon::circom

int main(int argc, char** argv) {
  return tachyon::circom::RealMain(argc, argv);
}
