#ifndef BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_
#define BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/poseidon/poseidon_config.h"
#include "benchmark/poseidon/simple_poseidon_benchmark_reporter.h"
// clang-format on
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"

namespace tachyon {

template <typename Field>
class PoseidonBenchmarkRunner {
 public:
  using CPrimeField = typename c::base::TypeTraits<Field>::CType;

  typedef CPrimeField* (*PoseidonExternalFn)(uint64_t* duration);

  PoseidonBenchmarkRunner(SimplePoseidonBenchmarkReporter* reporter,
                          PoseidonConfig* config)
      : reporter_(reporter), config_(config) {}

  Field Run() {
    Field ret;
    for (size_t i = 0; i < config_->repeating_num(); ++i) {
      crypto::PoseidonConfig<Field> config =
          crypto::PoseidonConfig<Field>::CreateCustom(8, 5, 8, 63, 0);
      crypto::PoseidonSponge<Field> sponge(config);
      crypto::SpongeState<Field> state(config);
      base::TimeTicks start = base::TimeTicks::Now();
      sponge.Permute(state);
      reporter_->AddTime(i, (base::TimeTicks::Now() - start).InSecondsF());
      if (i == 0) {
        ret = state.elements[1];
      }
    }
    return ret;
  }

  Field RunExternal(PoseidonExternalFn fn) {
    std::unique_ptr<CPrimeField> ret;
    for (size_t i = 0; i < config_->repeating_num(); ++i) {
      uint64_t duration_in_us;
      ret.reset(fn(&duration_in_us));
      reporter_->AddTime(i, base::Microseconds(duration_in_us).InSecondsF());
    }
    return *c::base::native_cast(ret.get());
  }

 private:
  // not owned
  SimplePoseidonBenchmarkReporter* const reporter_;
  // not owned
  PoseidonConfig* const config_;
};

}  // namespace tachyon

#endif  // BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_
