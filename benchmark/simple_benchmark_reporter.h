#ifndef BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_
#define BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_

#include <stddef.h>

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

  void AddTime(size_t idx, double time) { times_[idx].push_back(time); }

  void Show();

 protected:
  std::string title_;
  std::vector<std::string> column_headers_;
  std::vector<std::string> targets_;
  std::vector<std::vector<double>> times_;
};

}  // namespace tachyon

#endif  // BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_
