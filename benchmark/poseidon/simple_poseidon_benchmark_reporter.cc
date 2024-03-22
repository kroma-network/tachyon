#include "benchmark/poseidon/simple_poseidon_benchmark_reporter.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon {

SimplePoseidonBenchmarkReporter::SimplePoseidonBenchmarkReporter(
    std::string_view title, size_t repeating_num)
    : SimpleBenchmarkReporter(title) {
  targets_ = base::CreateVector(
      repeating_num, [](size_t i) { return base::NumberToString(i); });
  targets_.push_back("avg");

  AddVendor("tachyon");
  times_.resize(repeating_num + 1);
}

void SimplePoseidonBenchmarkReporter::AddVendor(std::string_view name) {
  column_headers_.push_back(std::string(name));
}

void SimplePoseidonBenchmarkReporter::AddAverageToLastRow() {
  for (size_t i = 0; i < column_headers_.size(); ++i) {
    double total = 0;
    for (size_t j = 0; j < targets_.size() - 1; ++j) {
      total += times_[j][i];
    }
    AddTime(targets_.size() - 1, total / (targets_.size() - 1));
  }
}

}  // namespace tachyon
