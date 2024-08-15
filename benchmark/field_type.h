#ifndef BENCHMARK_FIELD_TYPE_H_
#define BENCHMARK_FIELD_TYPE_H_

#include <stdint.h>

#include <string>
#include <string_view>

#include "tachyon/base/flag/flag_value_traits.h"
#include "tachyon/base/logging.h"

namespace tachyon {
namespace benchmark {

class FieldType {
 public:
  enum Value : uint32_t {
    // clang-format off
    kBabyBear  = 1 << 1,
    kBn254Fr   = 1 << 2,
    // clang-format on
  };

  constexpr static FieldType BabyBear() { return FieldType(kBabyBear); }
  constexpr static FieldType Bn254Fr() { return FieldType(kBn254Fr); }

  FieldType() = default;

  constexpr Value value() const { return value_; }
  constexpr bool operator==(FieldType a) const { return value_ == a.value_; }
  constexpr bool operator!=(FieldType a) const { return value_ != a.value_; }

  std::string_view ToString() const {
    switch (value_) {
      case FieldType::kBabyBear:
        return "baby_bear";
      case FieldType::kBn254Fr:
        return "bn254_fr";
    }
    NOTREACHED();
    return "";
  }

 private:
  explicit constexpr FieldType(Value v) : value_(v) {}

  Value value_;
};

}  // namespace benchmark

namespace base {

template <>
class FlagValueTraits<benchmark::FieldType> {
 public:
  using FieldType = benchmark::FieldType;

  static bool ParseValue(std::string_view input, FieldType* value,
                         std::string* reason) {
    if (input == "baby_bear") {
      *value = FieldType::BabyBear();
    } else if (input == "bn254_fr") {
      *value = FieldType::Bn254Fr();
    } else {
      *reason = absl::Substitute("Unknown prime field: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // BENCHMARK_FIELD_TYPE_H_
