#ifndef TACHYON_MATH_BASE_CONVERSIONS_H_
#define TACHYON_MATH_BASE_CONVERSIONS_H_

#include <string_view>

namespace tachyon {
namespace math {

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

template <size_t LimbNumbs>
class StringNumberConversion<BigInt<LimbNumbs>> {
 public:
  static BigInt<LimbNumbs> FromDecString(std::string_view str) {
    return BigInt<LimbNumbs>::FromDecString(str);
  }

  static BigInt<LimbNumbs> FromHexString(std::string_view str) {
    return BigInt<LimbNumbs>::FromHexString(str);
  }
};

template <size_t LimbNumbs>
class IntConversion<BigInt<LimbNumbs>> {
 public:
  template <typename T>
  static BigInt<LimbNumbs> FromInt(T v) {
    return BigInt<LimbNumbs>(v);
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_CONVERSIONS_H_
