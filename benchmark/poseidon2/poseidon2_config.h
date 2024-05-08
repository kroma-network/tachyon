#ifndef BENCHMARK_POSEIDON2_POSEIDON2_CONFIG_H_
#define BENCHMARK_POSEIDON2_POSEIDON2_CONFIG_H_

#include <stddef.h>

#include <string>
#include <vector>

namespace tachyon {

class Poseidon2Config {
 public:
  enum class PrimeField {
    kBn254Fr,
  };

  enum class Vendor {
    kHorizen,
    kPlonky3,
  };

  static std::string VendorToString(Vendor vendor);

  Poseidon2Config() = default;
  Poseidon2Config(const Poseidon2Config& other) = delete;
  Poseidon2Config& operator=(const Poseidon2Config& other) = delete;

  bool check_results() const { return check_results_; }
  size_t repeating_num() const { return repeating_num_; }
  PrimeField prime_field() const { return prime_field_; }
  const std::vector<Vendor>& vendors() const { return vendors_; }

  bool Parse(int argc, char** argv);

 private:
  bool check_results_ = false;
  size_t repeating_num_ = 10;
  PrimeField prime_field_;
  std::vector<Vendor> vendors_;
};

}  // namespace tachyon

#endif  // BENCHMARK_POSEIDON2_POSEIDON2_CONFIG_H_
