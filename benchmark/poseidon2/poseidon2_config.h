#ifndef BENCHMARK_POSEIDON2_POSEIDON2_CONFIG_H_
#define BENCHMARK_POSEIDON2_POSEIDON2_CONFIG_H_

#include <stddef.h>

#include <vector>

// clang-format off
#include "benchmark/config.h"
#include "benchmark/field_type.h"
// clang-format on

namespace tachyon::benchmark {

class Poseidon2Config : public Config {
 public:
  Poseidon2Config();
  Poseidon2Config(const Poseidon2Config& other) = delete;
  Poseidon2Config& operator=(const Poseidon2Config& other) = delete;

  size_t repeating_num() const { return repeating_num_; }
  FieldType prime_field() const { return prime_field_; }

 private:
  // Config methods
  bool Validate() const override;

  size_t repeating_num_;
  FieldType prime_field_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_POSEIDON2_POSEIDON2_CONFIG_H_
