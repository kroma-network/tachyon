#ifndef TACHYON_CRYPTO_HASHES_PRIME_FIELD_SERIALIZABLE_H_
#define TACHYON_CRYPTO_HASHES_PRIME_FIELD_SERIALIZABLE_H_

#include <stdint.h>

#include <numeric>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/math/base/big_int.h"
#include "tachyon/math/finite_fields/prime_field_forward.h"

namespace tachyon::crypto {

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
class PrimeFieldSerializable;

template <typename T>
class PrimeFieldSerializable<T, std::enable_if_t<std::is_integral_v<T>>> {
 public:
  template <typename Config>
  static bool ToPrimeField(const T& value, std::vector<math::PrimeField<Config>>* fields) {
    if (value >= Config::kModulus[0]) return false;
    fields->push_back(math::PrimeField<Config>(math::BigInt<1>(value)));
    return true;
  }

  template <typename PrimeFieldTy>
  static bool BatchToPrimeField(const absl::Span<const T>& values,
                                std::vector<PrimeFieldTy>* fields) {
    for (const T& value : values) {
      if (!PrimeFieldSerializable<T>::ToPrimeField(value, fields)) return false;
    }
    return true;
  }
};

template <typename T, typename PrimeFieldTy>
bool SerializeToFieldElements(const T& value,
                              std::vector<PrimeFieldTy>* fields) {
  return PrimeFieldSerializable<T>::ToPrimeField(value, fields);
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_BYTES_SERIALIZABLE_H_
