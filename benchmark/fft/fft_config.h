#ifndef BENCHMARK_FFT_FFT_CONFIG_H_
#define BENCHMARK_FFT_FFT_CONFIG_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace tachyon {

class FFTConfig {
 public:
  FFTConfig() = default;
  FFTConfig(const FFTConfig& other) = delete;
  FFTConfig& operator=(const FFTConfig& other) = delete;

  uint64_t k() const { return k_; }

  bool Parse(int argc, char** argv);

 private:
  uint64_t k_;
};

}  // namespace tachyon

#endif  // BENCHMARK_FFT_FFT_CONFIG_H_
