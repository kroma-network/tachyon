#ifndef BENCHMARK_SIMPLE_REPORTER_H_
#define BENCHMARK_SIMPLE_REPORTER_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/vendor.h"
// clang-format on
#include "tachyon/base/time/time.h"

namespace tachyon::benchmark {

class SimpleReporter {
 public:
  SimpleReporter() = default;
  SimpleReporter(const SimpleReporter& other) = delete;
  SimpleReporter& operator=(const SimpleReporter& other) = delete;

  void set_title(std::string_view title) { title_ = std::string(title); }

  void set_x_label(std::string_view x_label) {
    x_label_ = std::string(x_label);
  }

  void set_y_label(std::string_view y_label) {
    y_label_ = std::string(y_label);
  }

  void set_column_labels(const std::vector<std::string>& column_labels) {
    column_labels_ = column_labels;
  }

  void set_column_labels(std::vector<std::string>&& column_labels) {
    column_labels_ = std::move(column_labels);
  }

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
