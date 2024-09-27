#ifndef BENCHMARK_POSEIDON2_POSEIDON2_BENCHMARK_RUNNER_H_
#define BENCHMARK_POSEIDON2_POSEIDON2_BENCHMARK_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>

// clang-format off
#include "benchmark/simple_reporter.h"
#include "benchmark/poseidon2/poseidon2_config.h"
// clang-format on
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"

namespace tachyon::benchmark {

template <typename Field>
class Poseidon2BenchmarkRunner {
 public:
  using CPrimeField = typename c::base::TypeTraits<Field>::CType;

  typedef CPrimeField* (*PoseidonExternalFn)(uint64_t* duration);

  Poseidon2BenchmarkRunner(SimpleReporter& reporter,
                           const Poseidon2Config& config)
      : reporter_(reporter), config_(config) {}

  template <typename Params>
  Field Run() {
    reporter_.AddVendor(Vendor::Tachyon());
    Field ret = Field::Zero();
    for (size_t i = 0; i < config_.repeating_num(); ++i) {
      crypto::Poseidon2Sponge<Params> sponge;
      crypto::SpongeState<Params> state;
      base::TimeTicks start = base::TimeTicks::Now();
      for (size_t j = 0; j < 10000; ++j) {
        sponge.Permute(state);
      }
      reporter_.AddTime(Vendor::Tachyon(), base::TimeTicks::Now() - start);
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
  const Poseidon2Config& config_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_POSEIDON2_POSEIDON2_BENCHMARK_RUNNER_H_
