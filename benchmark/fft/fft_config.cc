#include "benchmark/fft/fft_config.h"

#include <set>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/ranges/algorithm.h"

namespace tachyon::benchmark {

FFTConfig::FFTConfig() : FFTConfig(Options()) {}

FFTConfig::FFTConfig(const Options& options)
    : Config(options), include_vendors_(options.include_vendors) {
  parser_.AddFlag<base::Flag<std::vector<uint32_t>>>(&exponents_)
      .set_short_name("-k")
      .set_required()
      .set_help(
          "Specify the exponent 'k's where the degree of poly to test is 2ᵏ.");
  parser_.AddFlag<base::BoolFlag>(&run_ifft_)
      .set_long_name("--run_ifft")
      .set_help("Run IFFT benchmark. Default is FFT benchmark.");
  if (include_vendors_) {
    parser_.AddFlag<base::Flag<std::set<Vendor>>>(&vendors_)
        .set_long_name("--vendor")
        .set_help(
            "Vendors to be benchmarked with. (supported vendors: arkworks, "
            "bellman, halo2)");
  }
}

void FFTConfig::PostParse() {
  base::ranges::sort(exponents_);  // NOLINT(build/include_what_you_use)
}

bool FFTConfig::Validate() const {
  if (include_vendors_) {
    for (const Vendor vendor : vendors_) {
      if ((vendor.value() != Vendor::kArkworks) &&
          (vendor.value() != Vendor::kBellman) &&
          (vendor.value() != Vendor::kScrollHalo2)) {
        tachyon_cerr << "Unsupported vendor " << vendor.ToString() << std::endl;
        return false;
      }
    }
  }
  return true;
}

std::vector<size_t> FFTConfig::GetDegrees() const {
  return base::Map(exponents_, [](uint32_t exponent) {
    return (size_t{1} << exponent) - 1;
  });
}

}  // namespace tachyon::benchmark
