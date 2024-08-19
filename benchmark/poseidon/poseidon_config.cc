#include "benchmark/poseidon/poseidon_config.h"

namespace tachyon::benchmark {

PoseidonConfig::PoseidonConfig() {
  parser_.AddFlag<base::Flag<size_t>>(&repeating_num_)
      .set_short_name("-n")
      .set_default_value(10)
      .set_help("Specify the number of repetition 'n'. By default, 10.");
}

}  // namespace tachyon::benchmark
