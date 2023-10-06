#include "benchmark/ec/ec_config.h"

#include <algorithm>
#include <string>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"

namespace tachyon {

bool ECConfig::Parse(int argc, char** argv) {
  base::FlagParser parser;
  parser.AddFlag<base::Flag<std::vector<uint64_t>>>(&point_nums_)
      .set_short_name("-n")
      .set_required()
      .set_help("The number of points to test");
  {
    std::string error;
    if (!parser.Parse(argc, argv, &error)) {
      tachyon_cerr << error << std::endl;
      return false;
    }
  }

  base::ranges::sort(point_nums_);
  return true;
}

}  // namespace tachyon
