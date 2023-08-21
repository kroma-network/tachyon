#ifndef BENCHMARK_MSM_MSM_CONFIG_H_
#define BENCHMARK_MSM_MSM_CONFIG_H_

#include <stdint.h>

#include <vector>

namespace tachyon {

class ECConfig {
 public:
  ECConfig() = default;
  ECConfig(const ECConfig& other) = delete;
  ECConfig& operator=(const ECConfig& other) = delete;

  const std::vector<uint64_t>& degrees() const { return degrees_; }

  bool Parse(int argc, char** argv);

  std::vector<uint64_t> GetPointNums() const;

 private:
  std::vector<uint64_t> degrees_;
};

}  // namespace tachyon

#endif  // BENCHMARK_MSM_MSM_CONFIG_H_
