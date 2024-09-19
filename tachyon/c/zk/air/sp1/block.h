#ifndef TACHYON_C_ZK_AIR_SP1_BLOCK_H_
#define TACHYON_C_ZK_AIR_SP1_BLOCK_H_

#include <array>
#include <string>
#include <type_traits>
#include <utility>

#include "tachyon/base/strings/string_util.h"
#include "tachyon/base/types/always_false.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

namespace tachyon::c::zk::air::sp1 {

template <typename T>
class Block {
 public:
  constexpr Block() = default;
  constexpr explicit Block(const std::array<T, 4>& value) : value_(value) {}
  constexpr explicit Block(std::array<T, 4>&& value)
      : value_(std::move(value)) {}

  template <typename F = T, std::enable_if_t<tachyon::math::FiniteFieldTraits<
                                F>::kIsFiniteField>* = nullptr>
  constexpr static Block From(const F& value) {
    if constexpr (tachyon::math::FiniteFieldTraits<F>::kIsPrimeField) {
      return Block({value, F::Zero(), F::Zero(), F::Zero()});
      // NOLINTNEXTLINE(readability/braces)
    } else if constexpr (tachyon::math::FiniteFieldTraits<
                             F>::kIsExtensionField) {
      return Block(value.ToBaseFields());
    } else {
      static_assert(tachyon::base::AlwaysFalse<F>);
    }
  }

  constexpr T& operator[](size_t idx) { return value_[idx]; }
  constexpr const T& operator[](size_t idx) const { return value_[idx]; }

  constexpr bool operator==(const Block& other) const {
    return value_ == other.value_;
  }
  constexpr bool operator!=(const Block& other) const {
    return value_ != other.value_;
  }

  std::string ToString() const {
    return tachyon::base::ContainerToString(value_);
  }

 private:
  std::array<T, 4> value_;
};

}  // namespace tachyon::c::zk::air::sp1

#endif  // TACHYON_C_ZK_AIR_SP1_BLOCK_H_
