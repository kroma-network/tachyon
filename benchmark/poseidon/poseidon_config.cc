#include "benchmark/poseidon/poseidon_config.h"

namespace tachyon::benchmark {

bool PoseidonConfig::Parse(int argc, char** argv) {
  parser_.AddFlag<base::Flag<size_t>>(&repeating_num_)
      .set_short_name("-n")
      .set_help("Specify the number of repetition 'n'. By default, 10.");

  return Config::Parse(
      argc, argv, {/*include_check_results=*/true, /*include_vendors=*/false});
}

}  // namespace tachyon::benchmark
