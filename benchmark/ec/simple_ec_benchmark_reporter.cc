#include "benchmark/ec/simple_ec_benchmark_reporter.h"

#include "absl/strings/substitute.h"

namespace tachyon {

SimpleECBenchmarkReporter::SimpleECBenchmarkReporter(
    const std::vector<uint64_t>& nums) {
  names_.reserve(nums.size() * 2);
  results_.reserve(nums.size() * 2);
  for (uint64_t num : nums) {
    names_.push_back(absl::Substitute("CPU/$0", num));
  }
  for (uint64_t num : nums) {
    names_.push_back(absl::Substitute("CUDA/$0", num));
  }
}

}  // namespace tachyon
