#ifndef BENCHMARK_FFT_BATCH_CONFIG_H_
#define BENCHMARK_FFT_BATCH_CONFIG_H_

#include <stddef.h>

#include <string>
#include <vector>

namespace tachyon {
class Config {
 public:
  enum class PrimeField {
    kBabyBear,
  };

  enum class Vendor {
    kPlonky3,
  };

  static std::string VendorToString(Vendor vendor);

  Config() = default;
  Config(const Config& other) = delete;
  Config& operator=(const Config& other) = delete;

  bool check_results() const { return check_results_; }
  size_t repeating_num() const { return repeating_num_; }
  size_t degree() const { return degree_; }
  size_t batch_size() const { return batch_size_; }
  PrimeField prime_field() const { return prime_field_; }
  const std::vector<Vendor>& vendors() const { return vendors_; }

  bool Parse(int argc, char** argv);

 private:
  bool check_results_ = false;
  size_t repeating_num_ = 10;
  uint32_t degree_ = 18;
  size_t batch_size_ = 100;
  PrimeField prime_field_;
  std::vector<Vendor> vendors_;
};

}  // namespace tachyon

#endif  // BENCHMARK_FFT_BATCH_CONFIG_H_
