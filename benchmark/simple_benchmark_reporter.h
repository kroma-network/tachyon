#ifndef BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_
#define BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_

#include <string>
#include <vector>

namespace tachyon {

class SimpleBenchmarkReporter {
 public:
  SimpleBenchmarkReporter() = default;
  explicit SimpleBenchmarkReporter(std::string_view title)
      : title_(std::string(title)) {}
  SimpleBenchmarkReporter(const SimpleBenchmarkReporter& other) = delete;
  SimpleBenchmarkReporter& operator=(const SimpleBenchmarkReporter& other) =
      delete;

  void AddResult(size_t idx, double result) { results_[idx].push_back(result); }

  void Show();

 protected:
  std::string title_;
  std::vector<std::string> column_headers_;
  std::vector<std::string> targets_;
  std::vector<std::vector<double>> results_;
};

}  // namespace tachyon

#endif  // BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_
