// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_PRIME_FIELD_SERIALIZABLE_H_
#define TACHYON_CRYPTO_HASHES_PRIME_FIELD_SERIALIZABLE_H_

#include <array>
#include <type_traits>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"

#include "tachyon/math/base/big_int.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

namespace tachyon::crypto {

template <typename T, typename SFINAE = void>
class PrimeFieldSerializable;

template <typename T>
class PrimeFieldSerializable<T, std::enable_if_t<std::is_integral_v<T>>> {
 public:
  template <typename PrimeField>
  constexpr static bool ToPrimeField(const T& value,
                                     std::vector<PrimeField>* fields) {
    if (math::BigInt<1>(value) >= PrimeField::Config::kModulus) return false;
    fields->push_back(PrimeField::FromBigInt(math::BigInt<1>(value)));
    return true;
  }

  template <typename PrimeField>
  constexpr static bool BatchToPrimeField(absl::Span<const T> values,
                                          std::vector<PrimeField>* fields) {
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

  constexpr static bool BatchToPrimeField(absl::Span<const T> values,
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
  template <typename PrimeField>
  constexpr static bool ToPrimeField(const std::vector<T>& values,
                                     std::vector<PrimeField>* fields) {
    return PrimeFieldSerializable<T>::BatchToPrimeField(
        absl::MakeConstSpan(values), fields);
  }
};

template <typename T, size_t N>
class PrimeFieldSerializable<absl::InlinedVector<T, N>> {
 public:
  template <typename PrimeField>
  constexpr static bool ToPrimeField(const absl::InlinedVector<T, N>& values,
                                     std::vector<PrimeField>* fields) {
    return PrimeFieldSerializable<T>::BatchToPrimeField(
        absl::MakeConstSpan(values), fields);
  }
};

template <typename T>
class PrimeFieldSerializable<absl::Span<T>> {
 public:
  template <typename PrimeField>
  constexpr static bool ToPrimeField(absl::Span<T> values,
                                     std::vector<PrimeField>* fields) {
    return PrimeFieldSerializable<std::remove_const_t<T>>::BatchToPrimeField(
        absl::MakeConstSpan(values), fields);
  }
};

template <typename T, size_t N>
class PrimeFieldSerializable<std::array<T, N>> {
 public:
  template <typename PrimeField>
  constexpr static bool ToPrimeField(const std::array<T, N>& values,
                                     std::vector<PrimeField>* fields) {
    return PrimeFieldSerializable<T>::BatchToPrimeField(
        absl::MakeConstSpan(values), fields);
  }
};

template <typename T, typename PrimeField>
constexpr bool SerializeToFieldElements(const T& value,
                                        std::vector<PrimeField>* fields) {
  return PrimeFieldSerializable<T>::ToPrimeField(value, fields);
}

template <typename T, typename PrimeField>
constexpr bool SerializeBatchToFieldElements(absl::Span<const T> values,
                                             std::vector<PrimeField>* fields) {
  return PrimeFieldSerializable<T>::BatchToPrimeField(values, fields);
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_PRIME_FIELD_SERIALIZABLE_H_
