#ifndef BENCHMARK_FFT_BATCH_FFT_BATCH_CONFIG_H_
#define BENCHMARK_FFT_BATCH_FFT_BATCH_CONFIG_H_

#include <stddef.h>

#include <vector>

// clang-format off
#include "benchmark/config.h"
#include "benchmark/field_type.h"
// clang-format on

namespace tachyon::benchmark {

class FFTBatchConfig : public Config {
 public:
  FFTBatchConfig();
  FFTBatchConfig(const FFTBatchConfig& other) = delete;
  FFTBatchConfig& operator=(const FFTBatchConfig& other) = delete;

  const std::vector<uint32_t>& exponents() const { return exponents_; }
  size_t batch_size() const { return batch_size_; }
  bool run_coset_lde() const { return run_coset_lde_; }
  FieldType prime_field() const { return prime_field_; }

  std::vector<size_t> GetDegrees() const;

 private:
  // Config methods
  void PostParse() override;
  bool Validate() const override;

  std::vector<uint32_t> exponents_;
  bool run_coset_lde_;
  size_t batch_size_;
  FieldType prime_field_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FFT_BATCH_FFT_BATCH_CONFIG_H_
