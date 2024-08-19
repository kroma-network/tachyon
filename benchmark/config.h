#ifndef BENCHMARK_CONFIG_H_
#define BENCHMARK_CONFIG_H_

#include <set>

// clang-format off
#include "benchmark/vendor.h"
// clang-format on
#include "tachyon/base/flag/flag_parser.h"

namespace tachyon::benchmark {

class Config {
 public:
  struct Options {
    bool include_check_results = false;
  };

  Config() = default;
  Config(const Config& other) = delete;
  Config& operator=(const Config& other) = delete;

  const std::set<Vendor>& vendors() const { return vendors_; }
  bool check_results() const { return check_results_; }

  bool Parse(int argc, char** argv, const Options& options);

 protected:
  base::FlagParser parser_;
  std::set<Vendor> vendors_;
  bool check_results_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_CONFIG_H_
