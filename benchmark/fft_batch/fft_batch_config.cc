#include "benchmark/fft_batch/fft_batch_config.h"

#include <algorithm>
#include <set>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/flag/flag_parser.h"

namespace tachyon::benchmark {

bool FFTBatchConfig::Parse(int argc, char** argv) {
  parser_.AddFlag<base::Flag<std::vector<uint32_t>>>(&exponents_)
      .set_short_name("-k")
      .set_required()
      .set_help(
          "Specify the exponent 'k's where the degree of poly to test is 2ᵏ.");
  parser_.AddFlag<base::BoolFlag>(&run_coset_lde_)
      .set_long_name("--run_coset_lde")
      .set_default_value(false)
      .set_help("Run CosetLDE benchmark. Default is FFT benchmark.");
  parser_.AddFlag<base::Flag<size_t>>(&batch_size_)
      .set_short_name("-b")
      .set_long_name("--batch_size")
      .set_default_value(100)
      .set_help("Specify the batch size. By default, 100.");
  parser_.AddFlag<base::Flag<FieldType>>(&prime_field_)
      .set_short_name("-p")
      .set_long_name("--prime_field")
      .set_default_value(FieldType::BabyBear())
      .set_help(
          "A prime field to be benchmarked with. (supported prime fields: "
          "baby_bear");
  parser_.AddFlag<base::Flag<std::set<Vendor>>>(&vendors_)
      .set_long_name("--vendor")
      .set_help("Vendors to be benchmarked with. (supported vendors: plonky3");

  if (!Config::Parse(
          argc, argv,
          {/*include_check_results=*/true, /*include_vendors=*/false})) {
    return false;
  }

  base::ranges::sort(exponents_);
  return true;
}

std::vector<size_t> FFTBatchConfig::GetDegrees() const {
  return base::Map(exponents_,
                   [](uint32_t exponent) { return (size_t{1} << exponent); });
}

}  // namespace tachyon::benchmark
