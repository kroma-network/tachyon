#ifndef BENCHMARK_SIMPLE_REPORTER_H_
#define BENCHMARK_SIMPLE_REPORTER_H_

#include <stddef.h>

#include <string>
#include <unordered_map>
#include <vector>

// clang-format off
#include "benchmark/vendor.h"
// clang-format on
#include "tachyon/base/time/time.h"

namespace tachyon::benchmark {

class SimpleReporter {
 public:
  SimpleReporter() = default;
  explicit SimpleReporter(std::string_view title)
      : title_(std::string(title)) {}
  SimpleReporter(const SimpleReporter& other) = delete;
  SimpleReporter& operator=(const SimpleReporter& other) = delete;

  void SetXLabel(std::string_view x_label);
  void SetYLabel(std::string_view x_label);
  void SetColumnLabels(const std::vector<std::string>& column_labels);
  void SetColumnLabels(std::vector<std::string>&& column_labels);

  void AddTime(Vendor vendor, base::TimeDelta time_taken);
  void AddVendor(Vendor name);
  void AddAverageAsLastColumn();

  void Show();
  void SetRepeatingNum(size_t repeating_num);

 protected:
  std::string title_;
  std::string x_label_;
  std::string y_label_ = "Time (s)";

  std::vector<Vendor> vendors_;
  std::vector<std::string> column_labels_;
  std::unordered_map<Vendor, std::vector<base::TimeDelta>> measurements_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_SIMPLE_REPORTER_H_
