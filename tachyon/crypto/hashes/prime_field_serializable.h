#ifndef TACHYON_CRYPTO_HASHES_PRIME_FIELD_SERIALIZABLE_H_
#define TACHYON_CRYPTO_HASHES_PRIME_FIELD_SERIALIZABLE_H_

#include <type_traits>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/math/base/big_int.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

namespace tachyon::crypto {

template <typename T, typename SFINAE = void>
class PrimeFieldSerializable;

template <typename T>
class PrimeFieldSerializable<T, std::enable_if_t<std::is_integral_v<T>>> {
 public:
  template <typename PrimeFieldTy>
  constexpr static bool ToPrimeField(const T& value,
                                     std::vector<PrimeFieldTy>* fields) {
    if (math::BigInt<1>(value) >= PrimeFieldTy::Config::kModulus) return false;
    fields->push_back(PrimeFieldTy::FromBigInt(math::BigInt<1>(value)));
    return true;
  }

  template <typename PrimeFieldTy>
  constexpr static bool BatchToPrimeField(const absl::Span<const T>& values,
                                          std::vector<PrimeFieldTy>* fields) {
    for (const T& value : values) {
      if (!PrimeFieldSerializable<T>::ToPrimeField(value, fields)) return false;
    }
    return true;
  }
};

template <typename T>
class PrimeFieldSerializable<
    T, std::enable_if_t<math::FiniteFieldTraits<T>::kIsPrimeField>> {
 public:
  constexpr static bool ToPrimeField(const T& value, std::vector<T>* fields) {
    fields->push_back(value);
    return true;
  }

  constexpr static bool BatchToPrimeField(const absl::Span<const T>& values,
                                          std::vector<T>* fields) {
    for (const T& value : values) {
      if (!PrimeFieldSerializable<T>::ToPrimeField(value, fields)) return false;
    }
    return true;
  }
};

template <typename T>
class PrimeFieldSerializable<std::vector<T>> {
 public:
  template <typename PrimeFieldTy>
  constexpr static bool ToPrimeField(const std::vector<T>& values,
                                     std::vector<PrimeFieldTy>* fields) {
    return PrimeFieldSerializable<T>::BatchToPrimeField(
        absl::MakeConstSpan(values), fields);
  }
};

template <typename T, typename PrimeFieldTy>
constexpr bool SerializeToFieldElements(const T& value,
                                        std::vector<PrimeFieldTy>* fields) {
  return PrimeFieldSerializable<T>::ToPrimeField(value, fields);
}

template <typename T, typename PrimeFieldTy>
constexpr bool SerializeBatchToFieldElements(
    const absl::Span<const T>& values, std::vector<PrimeFieldTy>* fields) {
  return PrimeFieldSerializable<T>::BatchToPrimeField(values, fields);
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_PRIME_FIELD_SERIALIZABLE_H_
