#include "benchmark/fft/simple_fft_benchmark_reporter.h"

#include <string>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon {

SimpleFFTBenchmarkReporter::SimpleFFTBenchmarkReporter(
    std::string_view title, const std::vector<uint64_t>& exponents)
    : SimpleBenchmarkReporter(title), exponents_(exponents) {
  targets_ = base::Map(exponents, [](uint64_t exponent) {
    return base::NumberToString(exponent);
  });
  results_.resize(exponents.size());
  AddVendor("tachyon");
}

void SimpleFFTBenchmarkReporter::AddVendor(std::string_view name) {
  column_headers_.push_back(std::string(name));
}

}  // namespace tachyon
