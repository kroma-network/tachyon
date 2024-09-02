#ifndef BENCHMARK_FRI_FRI_CONFIG_H_
#define BENCHMARK_FRI_FRI_CONFIG_H_

#include <stddef.h>

#include <vector>

// clang-format off
#include "benchmark/config.h"
// clang-format on

namespace tachyon::benchmark {

class FRIConfig : public Config {
 public:
  FRIConfig();
  FRIConfig(const FRIConfig& other) = delete;
  FRIConfig& operator=(const FRIConfig& other) = delete;

  const std::vector<uint32_t>& exponents() const { return exponents_; }
  size_t batch_size() const { return batch_size_; }
  size_t input_num() const { return input_num_; }
  uint32_t log_blowup() const { return log_blowup_; }

  std::vector<size_t> GetDegrees() const;

 private:
  // Config methods
  void PostParse() override;
  bool Validate() const override;

  std::vector<uint32_t> exponents_;
  size_t batch_size_;
  size_t input_num_;
  uint32_t log_blowup_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FRI_FRI_CONFIG_H_
