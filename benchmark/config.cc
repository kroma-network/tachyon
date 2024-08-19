#include "benchmark/config.h"

#include <string>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"

namespace tachyon::benchmark {

Config::Config() : Config(Options()) {}

Config::Config(const Options& options) {
  if (options.include_check_results) {
    parser_.AddFlag<base::BoolFlag>(&check_results_)
        .set_long_name("--check_results")
        .set_default_value(false)
        .set_help("Check results across different vendors. By default, false");
  }
}

bool Config::Parse(int argc, char** argv) {
  std::string error;
  if (!parser_.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return false;
  }

  PostParse();
  return true;
}

}  // namespace tachyon::benchmark
