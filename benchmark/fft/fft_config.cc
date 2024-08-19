#include "benchmark/fft/fft_config.h"

#include <set>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/ranges/algorithm.h"

namespace tachyon::benchmark {

bool FFTConfig::Parse(int argc, char** argv, const Options& options) {
  parser_.AddFlag<base::Flag<std::vector<uint32_t>>>(&exponents_)
      .set_short_name("-k")
      .set_required()
      .set_help(
          "Specify the exponent 'k's where the degree of poly to test is 2ᵏ.");
  parser_.AddFlag<base::BoolFlag>(&run_ifft_)
      .set_long_name("--run_ifft")
      .set_help("Run IFFT benchmark. Default is FFT benchmark.");
  parser_.AddFlag<base::Flag<std::set<Vendor>>>(&vendors_)
      .set_long_name("--vendor")
      .set_help(
          "Vendors to be benchmarked with. (supported vendors: arkworks, "
          "bellman, halo2)");

  if (!Config::Parse(
          argc, argv,
          {/*include_check_results=*/true, /*include_vendors=*/false})) {
    return false;
  }

  base::ranges::sort(exponents_);  // NOLINT
  return true;
}

std::vector<size_t> FFTConfig::GetDegrees() const {
  return base::Map(exponents_, [](uint32_t exponent) {
    return (size_t{1} << exponent) - 1;
  });
}

}  // namespace tachyon::benchmark
