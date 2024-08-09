#ifndef BENCHMARK_FFT_BATCH_BENCHMARK_REPORTER_H_
#define BENCHMARK_FFT_BATCH_BENCHMARK_REPORTER_H_

#include <string>

#include "benchmark/simple_benchmark_reporter.h"

namespace tachyon::benchmark::fft_batch {

class BenchmarkReporter : public SimpleBenchmarkReporter {
 public:
  BenchmarkReporter(std::string_view title, size_t repeating_num);
  BenchmarkReporter(const BenchmarkReporter& other) = delete;
  BenchmarkReporter& operator=(const BenchmarkReporter& other) = delete;

  void AddVendor(std::string_view name);

  void AddAverageToLastRow();
};

}  // namespace tachyon::benchmark::fft_batch

#endif  // BENCHMARK_FFT_BATCH_BENCHMARK_REPORTER_H_
