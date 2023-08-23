#ifndef BENCHMARK_SIMPLE_EC_BENCHMARK_REPORTER_H_
#define BENCHMARK_SIMPLE_EC_BENCHMARK_REPORTER_H_

#include "benchmark/simple_benchmark_reporter.h"

namespace tachyon {

class SimpleECBenchmarkReporter : public SimpleBenchmarkReporter {
 public:
  explicit SimpleECBenchmarkReporter(const std::vector<uint64_t>& nums);
  SimpleECBenchmarkReporter(const SimpleECBenchmarkReporter& other) = delete;
  SimpleECBenchmarkReporter& operator=(const SimpleECBenchmarkReporter& other) =
      delete;
};

}  // namespace tachyon

#endif  // BENCHMARK_SIMPLE_EC_BENCHMARK_REPORTER_H_
