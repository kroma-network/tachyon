#include "benchmark/config.h"

#include <string>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"

namespace tachyon::benchmark {

bool Config::Parse(int argc, char** argv, const Options& options) {
  // clang-format off
  if (options.include_check_results) {
    parser_.AddFlag<base::BoolFlag>(&check_results_)
        .set_long_name("--check_results")
        .set_default_value(false)
        .set_help("Check results across different vendors. By default, false");
  }
  if (options.include_vendors) {
    parser_.AddFlag<base::Flag<std::vector<Vendor>>>(&vendors_)
        .set_long_name("--vendor")
        .set_help("Vendors to be benchmarked with.");
  }
  // clang-format on

  {
    std::string error;
    if (!parser_.Parse(argc, argv, &error)) {
      tachyon_cerr << error << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace tachyon::benchmark
