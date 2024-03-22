#ifndef BENCHMARK_POSEIDON_SIMPLE_POSEIDON_BENCHMARK_REPORTER_H_
#define BENCHMARK_POSEIDON_SIMPLE_POSEIDON_BENCHMARK_REPORTER_H_

#include <string>

#include "benchmark/simple_benchmark_reporter.h"

namespace tachyon {

class SimplePoseidonBenchmarkReporter : public SimpleBenchmarkReporter {
 public:
  SimplePoseidonBenchmarkReporter(std::string_view title, size_t repeating_num);
  SimplePoseidonBenchmarkReporter(
      const SimplePoseidonBenchmarkReporter& other) = delete;
  SimplePoseidonBenchmarkReporter& operator=(
      const SimplePoseidonBenchmarkReporter& other) = delete;

  void AddVendor(std::string_view name);

  void AddAverageToLastRow();
};

}  // namespace tachyon

#endif  // BENCHMARK_POSEIDON_SIMPLE_POSEIDON_BENCHMARK_REPORTER_H_
