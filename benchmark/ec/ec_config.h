#ifndef BENCHMARK_EC_EC_CONFIG_H_
#define BENCHMARK_EC_EC_CONFIG_H_

#include <stdint.h>

#include <vector>

namespace tachyon {

class ECConfig {
 public:
  ECConfig() = default;
  ECConfig(const ECConfig& other) = delete;
  ECConfig& operator=(const ECConfig& other) = delete;

  const std::vector<uint64_t>& point_nums() const { return point_nums_; }

  bool Parse(int argc, char** argv);

 private:
  std::vector<uint64_t> point_nums_;
};

}  // namespace tachyon

#endif  // BENCHMARK_EC_EC_CONFIG_H_
