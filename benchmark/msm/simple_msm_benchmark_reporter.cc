#include "benchmark/msm/simple_msm_benchmark_reporter.h"

#include <string>
#include <vector>

#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon {

SimpleMSMBenchmarkReporter::SimpleMSMBenchmarkReporter(
    std::string_view title, const std::vector<uint64_t>& nums)
    : nums_(nums) {
  title_ = std::string(title);
  for (uint64_t num : nums) {
    targets_.push_back(base::NumberToString(num));
  }
  results_.resize(nums.size());
  AddVendor("tachyon");
}

void SimpleMSMBenchmarkReporter::AddVendor(std::string_view name) {
  column_headers_.push_back(std::string(name));
}

}  // namespace tachyon
