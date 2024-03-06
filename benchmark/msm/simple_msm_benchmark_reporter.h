#ifndef BENCHMARK_MSM_SIMPLE_MSM_BENCHMARK_REPORTER_H_
#define BENCHMARK_MSM_SIMPLE_MSM_BENCHMARK_REPORTER_H_

#include <vector>

#include "benchmark/simple_benchmark_reporter.h"

namespace tachyon {

class SimpleMSMBenchmarkReporter : public SimpleBenchmarkReporter {
 public:
  SimpleMSMBenchmarkReporter(std::string_view title,
                             const std::vector<uint64_t>& exponents);
  SimpleMSMBenchmarkReporter(const SimpleMSMBenchmarkReporter& other) = delete;
  SimpleMSMBenchmarkReporter& operator=(
      const SimpleMSMBenchmarkReporter& other) = delete;

  void AddVendor(std::string_view name);

 private:
  std::vector<uint64_t> exponents_;
};

}  // namespace tachyon

#endif  // BENCHMARK_MSM_SIMPLE_MSM_BENCHMARK_REPORTER_H_
