#include "benchmark/poseidon2/poseidon2_config.h"

#include <set>

#include "tachyon/base/console/iostream.h"

namespace tachyon::benchmark {

bool Poseidon2Config::Parse(int argc, char** argv) {
  parser_.AddFlag<base::Flag<size_t>>(&repeating_num_)
      .set_short_name("-n")
      .set_help("Specify the number of repetition 'n'. By default, 10.");
  parser_.AddFlag<base::Flag<FieldType>>(&prime_field_)
      .set_short_name("-p")
      .set_long_name("--prime_field")
      .set_help(
          "A prime field to be benchmarked with. (supported prime fields: "
          "bn254_fr)");
  parser_.AddFlag<base::Flag<std::set<Vendor>>>(&vendors_)
      .set_long_name("--vendor")
      .set_help(
          "Vendors to be benchmarked with. (supported vendors: horizen, "
          "plonky3)");

  return Config::Parse(
      argc, argv, {/*include_check_results=*/true, /*include_vendors=*/false});
}

}  // namespace tachyon::benchmark
