#ifndef BENCHMARK_FFT_FFT_CONFIG_H_
#define BENCHMARK_FFT_FFT_CONFIG_H_

#include <stdint.h>

#include <string>
#include <vector>

namespace tachyon {

class FFTConfig {
 public:
  enum class Vendor {
    kArkworks,
    kBellman,
    kHalo2,
  };

  static std::string VendorToString(Vendor vendor);

  FFTConfig() = default;
  FFTConfig(const FFTConfig& other) = delete;
  FFTConfig& operator=(const FFTConfig& other) = delete;

  const std::vector<size_t>& exponents() const { return exponents_; }
  const std::vector<Vendor>& vendors() const { return vendors_; }
  bool run_ifft() const { return run_ifft_; }
  bool check_results() const { return check_results_; }

  bool Parse(int argc, char** argv);

  std::vector<size_t> GetDegrees() const;

 private:
  std::vector<size_t> exponents_;
  std::vector<Vendor> vendors_;
  bool run_ifft_ = false;
  bool check_results_ = false;
};

}  // namespace tachyon

#endif  // BENCHMARK_FFT_FFT_CONFIG_H_
