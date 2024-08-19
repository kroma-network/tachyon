#include "benchmark/poseidon/poseidon_config.h"

namespace tachyon::benchmark {

PoseidonConfig::PoseidonConfig() : Config({/*include_check_results=*/true}) {
  parser_.AddFlag<base::Flag<size_t>>(&repeating_num_)
      .set_short_name("-n")
      .set_help("Specify the number of repetition 'n'. By default, 10.");
}

}  // namespace tachyon::benchmark
