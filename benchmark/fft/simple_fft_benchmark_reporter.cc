#include "benchmark/fft/simple_fft_benchmark_reporter.h"

#include <string>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon {

SimpleFFTBenchmarkReporter::SimpleFFTBenchmarkReporter(std::string_view title,
                                                       uint64_t k)
    : k_(k) {
  title_ = std::string(title);
  targets_.push_back(base::NumberToString(k));
  results_.resize(1);
  AddVendor("tachyon");
}

void SimpleFFTBenchmarkReporter::AddVendor(std::string_view name) {
  column_headers_.push_back(std::string(name));
}

}  // namespace tachyon
