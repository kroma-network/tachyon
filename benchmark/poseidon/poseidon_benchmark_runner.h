#ifndef BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_
#define BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/poseidon/poseidon_config.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"

namespace tachyon::benchmark {

template <typename Field>
class PoseidonBenchmarkRunner {
 public:
  using CPrimeField = typename c::base::TypeTraits<Field>::CType;

  typedef CPrimeField* (*PoseidonExternalFn)(uint64_t* duration);

  PoseidonBenchmarkRunner(SimpleReporter& reporter,
                          const PoseidonConfig& config)
      : reporter_(reporter), config_(config) {}

  Field Run() {
    reporter_.AddVendor(Vendor::TachyonCPU());
    Field ret;
    for (size_t i = 0; i < config_.repeating_num(); ++i) {
      crypto::PoseidonConfig<Field> config =
          crypto::PoseidonConfig<Field>::CreateCustom(8, 5, 8, 63, 0);
      crypto::PoseidonSponge<Field> sponge(std::move(config));
      crypto::SpongeState<Field> state(sponge.config);
      base::TimeTicks start = base::TimeTicks::Now();
      sponge.Permute(state);
      reporter_.AddTime(Vendor::TachyonCPU(), base::TimeTicks::Now() - start);
      if (i == 0) {
        ret = state.elements[1];
      }
    }
    return ret;
  }

  Field RunExternal(Vendor vendor, PoseidonExternalFn fn) {
    reporter_.AddVendor(vendor);
    std::unique_ptr<CPrimeField> ret;
    for (size_t i = 0; i < config_.repeating_num(); ++i) {
      uint64_t duration_in_us;
      ret.reset(fn(&duration_in_us));
      reporter_.AddTime(vendor, base::Microseconds(duration_in_us));
    }
    return *c::base::native_cast(ret.get());
  }

 private:
  SimpleReporter& reporter_;
  const PoseidonConfig& config_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_
