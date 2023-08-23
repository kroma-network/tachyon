#ifndef BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_
#define BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_

#include <string>
#include <vector>

namespace tachyon {

class SimpleBenchmarkReporter {
 public:
  SimpleBenchmarkReporter() = default;
  SimpleBenchmarkReporter(const std::vector<std::string>& names,
                          const std::vector<double>& results)
      : names_(names), results_(results) {}
  SimpleBenchmarkReporter(std::vector<std::string>&& names,
                          std::vector<double>&& results)
      : names_(std::move(names)), results_(std::move(results)) {}
  SimpleBenchmarkReporter(const SimpleBenchmarkReporter& other) = delete;
  SimpleBenchmarkReporter& operator=(const SimpleBenchmarkReporter& other) =
      delete;

  void AddResult(double result) { results_.push_back(result); }

  void Show();

 protected:
  std::vector<std::string> names_;
  std::vector<double> results_;
};

}  // namespace tachyon

#endif  // BENCHMARK_SIMPLE_BENCHMARK_REPORTER_H_
