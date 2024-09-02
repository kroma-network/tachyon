#include "benchmark/fri/fri_config.h"

#include <set>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"

namespace tachyon::benchmark {

FRIConfig::FRIConfig() {
  parser_.AddFlag<base::Flag<std::vector<uint32_t>>>(&exponents_)
      .set_short_name("-k")
      .set_required()
      .set_help(
          "Specify the exponent 'k's where the degree of poly to test is 2ᵏ.");
  parser_.AddFlag<base::Flag<size_t>>(&batch_size_)
      .set_short_name("-b")
      .set_long_name("--batch_size")
      .set_default_value(100)
      .set_help("Specify the batch size. By default, 100.");
  parser_.AddFlag<base::Flag<size_t>>(&input_num_)
      .set_short_name("-i")
      .set_long_name("--input_num")
      .set_default_value(4)
      .set_help(
          "Specify the number of inputs in a single round. By default, 4.");
  parser_.AddFlag<base::Flag<uint32_t>>(&log_blowup_)
      .set_short_name("-l")
      .set_long_name("--log_blowup")
      .set_default_value(1)
      .set_help("Specify the log blowup. By default, 1.");
  parser_.AddFlag<base::Flag<std::set<Vendor>>>(&vendors_)
      .set_long_name("--vendor")
      .set_help("Vendors to be benchmarked with. (supported vendors: plonky3");
}

void FRIConfig::PostParse() {
  base::ranges::sort(exponents_);  // NOLINT(build/include_what_you_use)
}

std::vector<size_t> FRIConfig::GetDegrees() const {
  return base::Map(exponents_,
                   [](uint32_t exponent) { return (size_t{1} << exponent); });
}

bool FRIConfig::Validate() const {
  for (const Vendor vendor : vendors_) {
    if (vendor.value() != Vendor::kPlonky3) {
      tachyon_cerr << "Unsupported vendor " << vendor.ToString() << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace tachyon::benchmark
