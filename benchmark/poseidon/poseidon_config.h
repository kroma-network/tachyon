#ifndef BENCHMARK_POSEIDON_POSEIDON_CONFIG_H_
#define BENCHMARK_POSEIDON_POSEIDON_CONFIG_H_

#include <stddef.h>

namespace tachyon {

class PoseidonConfig {
 public:
  PoseidonConfig() = default;
  PoseidonConfig(const PoseidonConfig& other) = delete;
  PoseidonConfig& operator=(const PoseidonConfig& other) = delete;

  bool check_results() const { return check_results_; }
  size_t repeating_num() const { return repeating_num_; }

  bool Parse(int argc, char** argv);

 private:
  bool check_results_ = false;
  size_t repeating_num_ = 10;
};

}  // namespace tachyon

#endif  // BENCHMARK_POSEIDON_POSEIDON_CONFIG_H_
