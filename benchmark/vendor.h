#ifndef BENCHMARK_VENDOR_H_
#define BENCHMARK_VENDOR_H_

#include <stdint.h>

#include <string>
#include <string_view>

#include "absl/strings/substitute.h"

#include "tachyon/base/flag/flag_value_traits.h"
#include "tachyon/base/logging.h"

namespace tachyon {
namespace benchmark {

class Vendor {
 public:
  enum Value : uint32_t {
    // clang-format off
    // Use |kTachyon| when comparing CPU results across vendors.
    kTachyon     = 1 << 0,
    // Use |kTachyonCPU| and |kTachyonGPU| when comparing CPU and GPU results.
    kTachyonCPU  = 1 << 1,
    kTachyonGPU  = 1 << 2,
    kArkworks    = 1 << 3,
    kBellman     = 1 << 4,
    kScrollHalo2 = 1 << 5,
    kPseHalo2    = 1 << 6,
    kHorizen     = 1 << 7,
    kPlonky3     = 1 << 8,
    // clang-format on
  };

  constexpr static Vendor Tachyon() { return Vendor(kTachyon); }
  constexpr static Vendor TachyonCPU() { return Vendor(kTachyonCPU); }
  constexpr static Vendor TachyonGPU() { return Vendor(kTachyonGPU); }
  constexpr static Vendor Arkworks() { return Vendor(kArkworks); }
  constexpr static Vendor Bellman() { return Vendor(kBellman); }
  constexpr static Vendor ScrollHalo2() { return Vendor(kScrollHalo2); }
  constexpr static Vendor PseHalo2() { return Vendor(kPseHalo2); }
  constexpr static Vendor Horizen() { return Vendor(kHorizen); }
  constexpr static Vendor Plonky3() { return Vendor(kPlonky3); }

  Vendor() = default;

  constexpr Value value() const { return value_; }

  constexpr bool operator==(Vendor a) const { return value_ == a.value_; }
  constexpr bool operator!=(Vendor a) const { return value_ != a.value_; }
  constexpr bool operator<(Vendor a) const { return value_ < a.value_; }

  std::string_view ToString() const {
    switch (value_) {
      case Vendor::kTachyon:
        return "tachyon";
      case Vendor::kTachyonCPU:
        return "tachyon_cpu";
      case Vendor::kTachyonGPU:
        return "tachyon_gpu";
      case Vendor::kArkworks:
        return "arkworks";
      case Vendor::kBellman:
        return "bellman";
      case Vendor::kScrollHalo2:
        return "halo2";
      case Vendor::kPseHalo2:
        return "pse_halo2";
      case Vendor::kHorizen:
        return "horizen";
      case Vendor::kPlonky3:
        return "plonky3";
    }
    NOTREACHED();
    return "";
  }

 private:
  explicit constexpr Vendor(Value v) : value_(v) {}

  Value value_;
};

}  // namespace benchmark

namespace base {

template <>
class FlagValueTraits<benchmark::Vendor> {
 public:
  using Vendor = benchmark::Vendor;

  static bool ParseValue(std::string_view input, Vendor* value,
                         std::string* reason) {
    if (input == "tachyon_cpu") {
      *value = Vendor::TachyonCPU();
    } else if (input == "tachyon_gpu") {
      *value = Vendor::TachyonGPU();
    } else if (input == "arkworks") {
      *value = Vendor::Arkworks();
    } else if (input == "bellman") {
      *value = Vendor::Bellman();
    } else if (input == "halo2") {
      *value = Vendor::ScrollHalo2();
    } else if (input == "pse_halo2") {
      *value = Vendor::PseHalo2();
    } else if (input == "horizen") {
      *value = Vendor::Horizen();
    } else if (input == "plonky3") {
      *value = Vendor::Plonky3();
    } else {
      *reason = absl::Substitute("Unknown vendor: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

template <>
struct std::hash<tachyon::benchmark::Vendor> {
  std::size_t operator()(const tachyon::benchmark::Vendor& v) const noexcept {
    return std::hash<uint32_t>{}(v.value());
  }
};

#endif  // BENCHMARK_VENDOR_H_
