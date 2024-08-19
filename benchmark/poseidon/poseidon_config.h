#ifndef BENCHMARK_POSEIDON_POSEIDON_CONFIG_H_
#define BENCHMARK_POSEIDON_POSEIDON_CONFIG_H_

#include <stddef.h>

// clang-format off
#include "benchmark/config.h"
// clang-format on

namespace tachyon::benchmark {

class PoseidonConfig : public Config {
 public:
  PoseidonConfig();
  PoseidonConfig(const PoseidonConfig& other) = delete;
  PoseidonConfig& operator=(const PoseidonConfig& other) = delete;

  size_t repeating_num() const { return repeating_num_; }

 private:
  size_t repeating_num_ = 10;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_POSEIDON_POSEIDON_CONFIG_H_
