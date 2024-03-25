#ifndef VENDORS_CIRCOM_BENCHMARK_RUNNER_H_
#define VENDORS_CIRCOM_BENCHMARK_RUNNER_H_

#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/files/file_path.h"
#include "tachyon/base/time/time.h"
#include "tachyon/zk/r1cs/groth16/proof.h"

namespace tachyon::circom {

template <typename Curve>
class Runner {
 public:
  using F = typename Curve::G1Curve::ScalarField;

  virtual ~Runner() = default;

  virtual void LoadZkey(const base::FilePath& zkey_path) = 0;
  virtual zk::r1cs::groth16::Proof<Curve> Run(
      const std::vector<F>& full_assignments, absl::Span<const F> public_inputs,
      base::TimeDelta& duration) = 0;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_BENCHMARK_RUNNER_H_
