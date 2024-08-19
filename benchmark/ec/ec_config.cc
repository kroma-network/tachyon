#include "benchmark/ec/ec_config.h"

#include "tachyon/base/ranges/algorithm.h"

namespace tachyon::benchmark {

ECConfig::ECConfig() : Config({/*include_check_results=*/false}) {
  parser_.AddFlag<base::Flag<std::vector<size_t>>>(&point_nums_)
      .set_short_name("-n")
      .set_required()
      .set_help("The number of points to test");
}

void ECConfig::PostParse() {
  base::ranges::sort(point_nums_);  // NOLINT(build/include_what_you_use)
}

}  // namespace tachyon::benchmark
