#include "benchmark/ec/ec_config.h"

#include <string>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/ranges/algorithm.h"

namespace tachyon::benchmark {

bool ECConfig::Parse(int argc, char** argv) {
  parser_.AddFlag<base::Flag<std::vector<uint64_t>>>(&point_nums_)
      .set_short_name("-n")
      .set_required()
      .set_help("The number of points to test");

  if (!Config::Parse(
          argc, argv,
          {/*include_check_results=*/false, /*include_vendors=*/false})) {
    return false;
  }

  base::ranges::sort(point_nums_);  // NOLINT
  return true;
}

}  // namespace tachyon::benchmark
