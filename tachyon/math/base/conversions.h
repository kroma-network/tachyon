#ifndef TACHYON_MATH_BASE_CONVERSIONS_H_
#define TACHYON_MATH_BASE_CONVERSIONS_H_

#include <string_view>

#include "tachyon/math/base/big_int.h"

namespace tachyon::math {

template <typename T, typename SFINAE = void>
class StringNumberConversion;

template <typename T>
T FromDecString(std::string_view str) {
  return StringNumberConversion<T>::FromDecString(str);
}

template <typename T>
T FromHexString(std::string_view str) {
  return StringNumberConversion<T>::FromHexString(str);
}

template <typename T, typename SFINAE = void>
class IntConversion;

template <typename R, typename T>
R FromInt(T v) {
  return IntConversion<R>::FromInt(v);
}

template <size_t N>
class StringNumberConversion<BigInt<N>> {
 public:
  static BigInt<N> FromDecString(std::string_view str) {
    return BigInt<N>::FromDecString(str);
  }

  static BigInt<N> FromHexString(std::string_view str) {
    return BigInt<N>::FromHexString(str);
  }
};

template <size_t N>
class IntConversion<BigInt<N>> {
 public:
  template <typename T>
  static BigInt<N> FromInt(T v) {
    return BigInt<N>(v);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_CONVERSIONS_H_
