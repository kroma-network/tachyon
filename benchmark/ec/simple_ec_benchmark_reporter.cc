#include "benchmark/ec/simple_ec_benchmark_reporter.h"

#include "absl/strings/substitute.h"

namespace tachyon {

SimpleECBenchmarkReporter::SimpleECBenchmarkReporter(
    std::string_view title, const std::vector<uint64_t>& nums) {
  title_ = std::string(title);
  column_headers_.push_back("CPU");
  column_headers_.push_back("GPU");
  for (uint64_t num : nums) {
    targets_.push_back(absl::StrCat(num));
  }
  results_.resize(nums.size());
}

}  // namespace tachyon
