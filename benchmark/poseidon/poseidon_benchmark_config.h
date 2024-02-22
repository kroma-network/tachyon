#ifndef BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_CONFIG_H_
#define BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_CONFIG_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/logging.h"

namespace tachyon::crypto {

class PoseidonBenchmarkConfig {
 public:
  PoseidonBenchmarkConfig() = default;
  PoseidonBenchmarkConfig(const PoseidonBenchmarkConfig& other) = delete;
  PoseidonBenchmarkConfig& operator=(const PoseidonBenchmarkConfig& other) =
      delete;

  bool check_results() const { return check_results_; }
  size_t repeating_num() const { return repeating_num_; }
  size_t absorbing_num() const { return absorbing_num_; }
  size_t squeezing_num() const { return squeezing_num_; }

  bool Parse(int argc, char** argv) {
    base::FlagParser parser;

    // NOTE(dongchangYoo): Parameters not entered among |n|, |a| and |s| will
    // use default values.
    parser.AddFlag<base::Flag<size_t>>(&repeating_num_)
        .set_short_name("-n")
        .set_help("Specify the number of repeatation 'n'.");
    parser.AddFlag<base::Flag<size_t>>(&absorbing_num_)
        .set_short_name("-a")
        .set_help("Specify the number of absorptions 'a'.");
    parser.AddFlag<base::Flag<size_t>>(&squeezing_num_)
        .set_short_name("-s")
        .set_help("Specify the number of squeeze 's'.");

    {
      std::string error;
      if (!parser.Parse(argc, argv, &error)) {
        tachyon_cerr << error << std::endl;
        return false;
      }
    }

    return true;
  }

 private:
  bool check_results_ = false;
  size_t repeating_num_ = 10;
  size_t absorbing_num_ = 5;
  size_t squeezing_num_ = 2;
};

}  // namespace tachyon::crypto

#endif  // BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_CONFIG_H_
