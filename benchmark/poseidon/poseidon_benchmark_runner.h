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
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_traits.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"

namespace tachyon {

template <typename Field>
class PoseidonBenchmarkRunner {
 public:
  using CPrimeField = typename c::base::TypeTraits<Field>::CType;

  typedef CPrimeField* (*PoseidonExternalFn)(const CPrimeField* pre_images,
                                             size_t absorbing_num,
                                             size_t squeezing_num,
                                             uint64_t* duration);

  PoseidonBenchmarkRunner(SimplePoseidonBenchmarkReporter* reporter,
                          PoseidonConfig* config)
      : reporter_(reporter), config_(config) {
    pre_images_ = base::CreateVector(config->absorbing_num(),
                                     []() { return Field::Random(); });
  }

  void Run(std::vector<Field>* results) {
    for (size_t i = 0; i < config_->repeating_num(); ++i) {
      crypto::PoseidonConfig<Field> config =
          crypto::PoseidonConfig<Field>::CreateCustom(8, 5, 8, 63, 0);
      crypto::PoseidonSponge<Field> sponge(config);
      base::TimeTicks start = base::TimeTicks::Now();
      for (size_t j = 0; j < config_->absorbing_num(); ++j) {
        sponge.Absorb(pre_images_[j]);
      }
      std::vector<Field> squeezed_elements =
          sponge.SqueezeFieldElements(config_->squeezing_num());
      reporter_->AddTime(i, (base::TimeTicks::Now() - start).InSecondsF());
      results->push_back(std::move(squeezed_elements[0]));
    }
  }

  void RunExternal(PoseidonExternalFn fn, std::vector<Field>* results) {
    for (size_t i = 0; i < config_->repeating_num(); ++i) {
      std::unique_ptr<CPrimeField> last_squeezed_element;
      uint64_t duration_in_us;
      last_squeezed_element.reset(fn(
          reinterpret_cast<const CPrimeField*>(pre_images_.data()),
          config_->absorbing_num(), config_->squeezing_num(), &duration_in_us));
      reporter_->AddTime(i, base::Microseconds(duration_in_us).InSecondsF());
      results->push_back(
          *reinterpret_cast<Field*>(last_squeezed_element.get()));
    }
  }

 private:
  // not owned
  SimplePoseidonBenchmarkReporter* const reporter_;
  // not owned
  PoseidonConfig* const config_;
  std::vector<Field> pre_images_;
};

}  // namespace tachyon

#endif  // BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_
