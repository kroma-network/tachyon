#include "benchmark/msm/simple_msm_benchmark_reporter.h"

#include "absl/strings/substitute.h"

namespace tachyon {

SimpleMSMBenchmarkReporter::SimpleMSMBenchmarkReporter(
    const std::vector<uint64_t>& nums)
    : nums_(nums) {
  AddVendor("tachyon");
}

void SimpleMSMBenchmarkReporter::AddVendor(std::string_view name) {
  names_.reserve(names_.size() + nums_.size());
  results_.reserve(results_.size() + nums_.size());
  for (uint64_t num : nums_) {
    names_.push_back(absl::Substitute("$0/$1", name, num));
  }
}

}  // namespace tachyon
