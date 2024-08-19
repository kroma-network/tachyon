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
    bool include_check_results = true;
  };

  Config();
  explicit Config(const Options& options);
  Config(const Config& other) = delete;
  Config& operator=(const Config& other) = delete;
  virtual ~Config() = default;

  const std::set<Vendor>& vendors() const { return vendors_; }
  bool check_results() const { return check_results_; }

  bool Parse(int argc, char** argv);

  virtual bool Validate() const { return true; }

 protected:
  // Override this method if you need to perform any actions after |Parse()| is
  // called.
  virtual void PostParse() {}

  base::FlagParser parser_;
  std::set<Vendor> vendors_;
  bool check_results_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_CONFIG_H_
