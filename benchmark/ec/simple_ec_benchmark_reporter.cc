#include "benchmark/ec/simple_ec_benchmark_reporter.h"

#include <string>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon {

SimpleECBenchmarkReporter::SimpleECBenchmarkReporter(
    std::string_view title, const std::vector<uint64_t>& nums)
    : SimpleBenchmarkReporter(title) {
  column_headers_.push_back("CPU");
  column_headers_.push_back("GPU");
  targets_ =
      base::Map(nums, [](uint64_t num) { return base::NumberToString(num); });
  results_.resize(nums.size());
}

}  // namespace tachyon
