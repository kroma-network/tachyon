#ifndef BENCHMARK_EC_EC_CONFIG_H_
#define BENCHMARK_EC_EC_CONFIG_H_

#include <vector>

// clang-format off
#include "benchmark/config.h"
// clang-format on

namespace tachyon::benchmark {

class ECConfig : public Config {
 public:
  ECConfig();
  ECConfig(const ECConfig& other) = delete;
  ECConfig& operator=(const ECConfig& other) = delete;

  const std::vector<size_t>& point_nums() const { return point_nums_; }

 private:
  // Config methods
  void PostParse() override;

  std::vector<size_t> point_nums_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_EC_EC_CONFIG_H_
