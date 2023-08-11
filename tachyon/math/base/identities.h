#ifndef TACHYON_MATH_BASE_IDENTITIES_H_
#define TACHYON_MATH_BASE_IDENTITIES_H_

namespace tachyon::math {

template <typename T, typename SFINAE = void>
class MultiplicativeIdentity;

template <typename T>
const T& One() {
  return MultiplicativeIdentity<T>::One();
}

template <typename T>
constexpr bool IsOne(const T& value) {
  return MultiplicativeIdentity<T>::IsOne(value);
}

template <typename T, typename SFINAE = void>
class AdditiveIdentity;

template <typename T>
const T& Zero() {
  return AdditiveIdentity<T>::Zero();
}

template <typename T>
constexpr bool IsZero(const T& value) {
  return AdditiveIdentity<T>::IsZero(value);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_IDENTITIES_H_
