#ifndef BENCHMARK_MSM_MSM_CONFIG_H_
#define BENCHMARK_MSM_MSM_CONFIG_H_

#include <stdint.h>

#include <vector>

namespace tachyon {

class MSMConfig {
 public:
  MSMConfig() = default;
  MSMConfig(const MSMConfig& other) = delete;
  MSMConfig& operator=(const MSMConfig& other) = delete;

  const std::vector<uint64_t>& degrees() const { return degrees_; }

  bool Parse(int argc, char** argv);

  std::vector<uint64_t> GetPointNums() const;

 private:
  std::vector<uint64_t> degrees_;
};

}  // namespace tachyon

#endif  // BENCHMARK_MSM_MSM_CONFIG_H_
