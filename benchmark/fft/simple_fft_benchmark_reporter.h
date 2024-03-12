#ifndef BENCHMARK_FFT_SIMPLE_FFT_BENCHMARK_REPORTER_H_
#define BENCHMARK_FFT_SIMPLE_FFT_BENCHMARK_REPORTER_H_

#include <vector>

#include "benchmark/simple_benchmark_reporter.h"

namespace tachyon {

class SimpleFFTBenchmarkReporter : public SimpleBenchmarkReporter {
 public:
  SimpleFFTBenchmarkReporter(std::string_view title,
                             const std::vector<uint64_t>& exponents);
  SimpleFFTBenchmarkReporter(const SimpleFFTBenchmarkReporter& other) = delete;
  SimpleFFTBenchmarkReporter& operator=(
      const SimpleFFTBenchmarkReporter& other) = delete;

  void AddVendor(std::string_view name);

 private:
  std::vector<uint64_t> exponents_;
};

}  // namespace tachyon

#endif  // BENCHMARK_FFT_SIMPLE_FFT_BENCHMARK_REPORTER_H_
