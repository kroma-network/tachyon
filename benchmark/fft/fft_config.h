#ifndef BENCHMARK_FFT_FFT_CONFIG_H_
#define BENCHMARK_FFT_FFT_CONFIG_H_

#include <stdint.h>

#include <string>
#include <vector>

// clang-format off
#include "benchmark/config.h"
// clang-format on

namespace tachyon::benchmark {

class FFTConfig : public Config {
 public:
  FFTConfig() = default;
  FFTConfig(const FFTConfig& other) = delete;
  FFTConfig& operator=(const FFTConfig& other) = delete;

  const std::vector<uint32_t>& exponents() const { return exponents_; }
  bool run_ifft() const { return run_ifft_; }

  bool Parse(int argc, char** argv, const Options& options);

  std::vector<size_t> GetDegrees() const;

 private:
  std::vector<uint32_t> exponents_;
  bool run_ifft_ = false;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FFT_FFT_CONFIG_H_
