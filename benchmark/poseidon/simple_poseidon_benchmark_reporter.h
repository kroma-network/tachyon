#ifndef BENCHMARK_POSEIDON_SIMPLE_POSEIDON_BENCHMARK_REPORTER_H_
#define BENCHMARK_POSEIDON_SIMPLE_POSEIDON_BENCHMARK_REPORTER_H_

#include <string>
#include <vector>

#include "benchmark/simple_benchmark_reporter.h"

#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon {

class SimplePoseidonBenchmarkReporter : public SimpleBenchmarkReporter {
 public:
  SimplePoseidonBenchmarkReporter(std::string_view title,
                                  size_t repeating_num) {
    title_ = std::string(title);
    for (size_t i = 0; i < repeating_num; ++i) {
      targets_.push_back(base::NumberToString(i));
    }
    targets_.push_back("avg");

    AddVendor("tachyon");
    times_.resize(repeating_num + 1);
  }

  SimplePoseidonBenchmarkReporter(
      const SimplePoseidonBenchmarkReporter& other) = delete;
  SimplePoseidonBenchmarkReporter& operator=(
      const SimplePoseidonBenchmarkReporter& other) = delete;

  void AddVendor(std::string_view name) {
    column_headers_.push_back(std::string(name));
  }

  std::string_view GetVendorName(size_t i) const { return column_headers_[i]; }

  void AddAverageToLastRow() {
    for (size_t i = 0; i < column_headers_.size(); ++i) {
      double total = 0;
      for (size_t j = 0; j < targets_.size() - 1; ++j) {
        total += times_[j][i];
      }
      AddTime(targets_.size() - 1, total / (targets_.size() - 1));
    }
  }

  size_t GetVendorNum() const { return column_headers_.size(); }
};

}  // namespace tachyon

#endif  // BENCHMARK_POSEIDON_SIMPLE_POSEIDON_BENCHMARK_REPORTER_H_
