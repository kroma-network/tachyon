#ifndef BENCHMARK_FFT_FFT_CONFIG_H_
#define BENCHMARK_FFT_FFT_CONFIG_H_

#include <stdint.h>

#include <vector>

// clang-format off
#include "benchmark/config.h"
// clang-format on

namespace tachyon::benchmark {

class FFTConfig : public Config {
 public:
  struct Options : public Config::Options {
    bool include_vendors = false;
  };

  FFTConfig();
  explicit FFTConfig(const Options& options);
  FFTConfig(const FFTConfig& other) = delete;
  FFTConfig& operator=(const FFTConfig& other) = delete;

  const std::vector<uint32_t>& exponents() const { return exponents_; }
  bool run_ifft() const { return run_ifft_; }

  std::vector<size_t> GetDegrees() const;

 private:
  // Config methods
  void PostParse() override;
  bool Validate() const override;

  std::vector<uint32_t> exponents_;
  bool run_ifft_;
  bool include_vendors_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FFT_FFT_CONFIG_H_
