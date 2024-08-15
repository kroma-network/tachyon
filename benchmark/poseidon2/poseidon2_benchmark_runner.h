#ifndef BENCHMARK_POSEIDON2_POSEIDON2_BENCHMARK_RUNNER_H_
#define BENCHMARK_POSEIDON2_POSEIDON2_BENCHMARK_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/poseidon/simple_poseidon_benchmark_reporter.h"
#include "benchmark/poseidon2/poseidon2_config.h"
// clang-format on
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_external_matrix.h"

namespace tachyon {

template <typename Field>
class Poseidon2BenchmarkRunner {
 public:
  using CPrimeField = typename c::base::TypeTraits<Field>::CType;

  typedef CPrimeField* (*PoseidonExternalFn)(uint64_t* duration);

  Poseidon2BenchmarkRunner(SimplePoseidonBenchmarkReporter& reporter,
                           const Poseidon2Config& config)
      : reporter_(reporter), config_(config) {}

  Field Run(const crypto::Poseidon2Config<Field>& config) {
    Field ret = Field::Zero();
    for (size_t i = 0; i < config_.repeating_num(); ++i) {
      crypto::Poseidon2Sponge<crypto::Poseidon2ExternalMatrix<
          crypto::Poseidon2HorizenExternalMatrix<Field>>>
          sponge(config);
      crypto::SpongeState<Field> state(config);
      base::TimeTicks start = base::TimeTicks::Now();
      for (size_t j = 0; j < 100; ++j) {
        sponge.Permute(state);
      }
      reporter_.AddTime(i, (base::TimeTicks::Now() - start).InSecondsF());
      if (i == 0) {
        ret = state.elements[1];
      }
    }
    return ret;
  }

  Field RunExternal(PoseidonExternalFn fn) {
    std::unique_ptr<CPrimeField> ret;
    for (size_t i = 0; i < config_.repeating_num(); ++i) {
      uint64_t duration_in_us;
      ret.reset(fn(&duration_in_us));
      reporter_.AddTime(i, base::Microseconds(duration_in_us).InSecondsF());
    }
    return *c::base::native_cast(ret.get());
  }

 private:
  SimplePoseidonBenchmarkReporter& reporter_;
  const Poseidon2Config& config_;
};

}  // namespace tachyon

#endif  // BENCHMARK_POSEIDON2_POSEIDON2_BENCHMARK_RUNNER_H_
