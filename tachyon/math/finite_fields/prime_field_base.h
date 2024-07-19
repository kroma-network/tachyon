// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <string>
#include <utility>

#include "absl/hash/hash.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/json/json.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/finite_fields/finite_field.h"
#include "tachyon/math/finite_fields/legendre_symbol.h"
#include "tachyon/math/finite_fields/packed_prime_field_traits_forward.h"
#include "tachyon/math/finite_fields/prime_field_util.h"

namespace tachyon {
namespace math {

struct TACHYON_EXPORT PrimeFieldFactors {
  uint32_t q_adicity;
  uint64_t q_part;
  uint32_t two_adicity;
  uint64_t two_part;
};

// PrimeField is a finite field GF(p) for p is prime.
// See https://mathworld.wolfram.com/PrimeField.html
template <typename F>
class PrimeFieldBase : public FiniteField<F> {
 public:
  using Config = typename FiniteFieldTraits<F>::Config;

  constexpr static bool HasRootOfUnity() {
    return Config::kHasTwoAdicRootOfUnity ||
           Config::kHasLargeSubgroupRootOfUnity;
  }

  // An invariant of a field which is prime number N
  // such that N * e(unit element) = 0.
  // It is uniquely determined for a given field.
  // See https://encyclopediaofmath.org/wiki/Characteristic_of_a_field
  constexpr static F Characteristic() {
    return F::FromBigint(Config::kModulus);
  }

  constexpr static uint32_t ExtensionDegree() { return 1; }

  constexpr static bool Decompose(uint64_t n, PrimeFieldFactors* factors) {
    static_assert(Config::kHasLargeSubgroupRootOfUnity);

    // Compute the size of our evaluation domain
    uint64_t q = uint64_t{F::Config::kSmallSubgroupBase};
    uint32_t q_adicity = ComputeAdicity(q, gmp::FromUnsignedInt(n));
    uint64_t q_part = static_cast<uint64_t>(std::pow(q, q_adicity));

    uint32_t two_adicity = ComputeAdicity(2, gmp::FromUnsignedInt(n));
    uint64_t two_part = static_cast<uint64_t>(std::pow(2, two_adicity));
    if (n != q_part * two_part) return false;

    factors->q_adicity = q_adicity;
    factors->q_part = q_part;
    factors->two_adicity = two_adicity;
    factors->two_part = two_part;
    return true;
  }

  // Returns false for either of the following cases:
  //
  // When there exists |Config::kLargeSubgroupRootOfUnity|:
  //   1. n is not a power of 2 times a power of |Config::kSmallSubgroupBase|.
  //   2. two-adicity of n is greater than |Config::kTwoAdicity|.
  //
  // When |Config::kLargeSubgroupRootOfUnity| does not exist:
  //   1. n is not a power of 2.
  //   2. two-adicity of next power of 2 of n is greater than
  //   |Config::kTwoAdicity|.
  static bool GetRootOfUnity(uint64_t n, F* ret) {
    static_assert(HasRootOfUnity());
    F omega;
    if constexpr (Config::kHasLargeSubgroupRootOfUnity) {
      PrimeFieldFactors factors;
      if (!Decompose(n, &factors)) return false;
      if (factors.two_adicity > Config::kTwoAdicity ||
          factors.q_adicity > Config::kSmallSubgroupAdicity) {
        return false;
      }

      if constexpr (F::Config::kUseMontgomery) {
        omega = F::FromMontgomery(Config::kLargeSubgroupRootOfUnity);
      } else {
        omega = F(Config::kLargeSubgroupRootOfUnity);
      }
      for (size_t i = factors.q_adicity; i < Config::kSmallSubgroupAdicity;
           ++i) {
        omega = omega.Pow(Config::kSmallSubgroupBase);
      }

      for (size_t i = factors.two_adicity; i < Config::kTwoAdicity; ++i) {
        omega.SquareInPlace();
      }
    } else {
      uint32_t log_size_of_group = base::bits::Log2Ceiling(n);
      uint64_t size = uint64_t{1} << log_size_of_group;

      if (n != size || log_size_of_group > Config::kTwoAdicity) {
        return false;
      }

      if constexpr (F::Config::kUseMontgomery) {
        omega = F::FromMontgomery(Config::kTwoAdicRootOfUnity);
      } else {
        omega = F(Config::kTwoAdicRootOfUnity);
      }
      for (uint32_t i = log_size_of_group; i < Config::kTwoAdicity; ++i) {
        omega.SquareInPlace();
      }
    }
    *ret = omega;
    return true;
  }

  constexpr LegendreSymbol Legendre() const {
    const F* f = static_cast<const F*>(this);
    // s = a^((p - 1) / 2)
    F s = f->Pow(Config::kModulusMinusOneDivTwo);
    if (s.IsZero())
      return LegendreSymbol::kZero;
    else if (s.IsOne())
      return LegendreSymbol::kOne;
    return LegendreSymbol::kMinusOne;
  }

  constexpr F& FrobeniusMapInPlace(uint32_t exponent) {
    // Do nothing.
    return static_cast<F&>(*this);
  }
};

template <typename H, typename F,
          std::enable_if_t<std::is_base_of_v<PrimeFieldBase<F>, F>>* = nullptr>
H AbslHashValue(H h, const F& prime_field) {
  if constexpr (F::Config::kModulusBits > 32) {
    for (uint64_t limb : prime_field.value().limbs) {
      h = H::combine(std::move(h), limb);
    }
  } else {
    h = H::combine(std::move(h), prime_field.value());
  }
  return h;
}

// NOTE(ashjeong): Base prime fields with no packed field version have their
// packed prime field type defaulted as themselves to ensure prime field
// standardization. See how it is used in
// tachyon/math/polynomials/univariate/radix2_evaluation_domain.h.
template <typename T>
struct PackedPrimeFieldTraits<
    T, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>, T>>> {
  using PackedPrimeField = T;
};

template <typename T>
struct ExtensionFieldTraits<
    T, std::enable_if_t<std::is_base_of_v<PrimeFieldBase<T>, T>>> {
  constexpr static uint32_t kDegreeOverBaseField = 1;
  constexpr static uint32_t kDegreeOverBasePrimeField = 1;

  using BaseField = T;
  using BasePrimeField = T;
};

}  // namespace math

namespace base {

template <typename T>
class Copyable<
    T, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>, T>>> {
 public:
  using value_type = typename T::value_type;
  using BigInt = typename T::BigIntTy;

  static bool s_allow_value_greater_than_or_equal_to_modulus;
  static bool s_is_in_montgomery;

  static bool WriteTo(const T& prime_field, Buffer* buffer) {
    if constexpr (T::Config::kUseMontgomery) {
      if (s_is_in_montgomery) {
        return buffer->Write(prime_field.value());
      }
    }
    if constexpr (T::Config::kModulusBits <= 32) {
      if constexpr (T::Config::kUseMontgomery) {
        return buffer->Write(T::Config::FromMontgomery(prime_field.value()));
      } else {
        return buffer->Write(prime_field.value());
      }
    } else {
      return buffer->Write(prime_field.ToBigInt());
    }
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, T* prime_field) {
    value_type v;
    if (!buffer.Read(&v)) return false;
    if (s_allow_value_greater_than_or_equal_to_modulus) {
      if (v >= T::Config::kModulus) {
        v %= T::Config::kModulus;
      }
    }
    if constexpr (T::Config::kUseMontgomery) {
      if (s_is_in_montgomery) {
        *prime_field = T::FromMontgomery(v);
        return true;
      }
    }
    *prime_field = T(v);
    return true;
  }

  static size_t EstimateSize(const T& prime_field) {
    if constexpr (T::Config::kModulusBits <= 32) {
      return sizeof(uint32_t);
    } else {
      return BigInt::kLimbNums * sizeof(uint64_t);
    }
  }
};

// static
template <typename T>
bool Copyable<T,
              std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>, T>>>::
    s_allow_value_greater_than_or_equal_to_modulus = false;

// static
template <typename T>
bool Copyable<T, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>,
                                                    T>>>::s_is_in_montgomery =
    false;

template <typename T>
class RapidJsonValueConverter<
    T, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>, T>>> {
 public:
  using value_type = typename T::value_type;
  using BigInt = typename T::BigIntTy;

  static bool s_allow_value_greater_than_or_equal_to_modulus;
  static bool s_is_in_montgomery;

  template <typename Allocator>
  static rapidjson::Value From(const T& value, Allocator& allocator) {
    if constexpr (T::Config::kUseMontgomery) {
      if (s_is_in_montgomery) {
        return RapidJsonValueConverter<value_type>::From(value.value(),
                                                         allocator);
      }
    }
    if constexpr (T::Config::kModulusBits <= 32) {
      if constexpr (T::Config::kUseMontgomery) {
        return RapidJsonValueConverter<uint32_t>::From(
            T::Config::FromMontgomery(value.value()), allocator);
      } else {
        return RapidJsonValueConverter<uint32_t>::From(value.value(),
                                                       allocator);
      }
    } else {
      return RapidJsonValueConverter<BigInt>::From(value.ToBigInt(), allocator);
    }
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 T* value, std::string* error) {
    value_type v;
    if (!RapidJsonValueConverter<value_type>::To(json_value, key, &v, error))
      return false;

    if (s_allow_value_greater_than_or_equal_to_modulus) {
      if (v >= T::Config::kModulus) {
        v %= T::Config::kModulus;
      }
    }
    if constexpr (T::Config::kUseMontgomery) {
      if (s_is_in_montgomery) {
        *value = T::FromMontgomery(v);
        return true;
      }
    }
    *value = T(v);
    return true;
  }
};

// static
template <typename T>
bool RapidJsonValueConverter<
    T, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>, T>>>::
    s_allow_value_greater_than_or_equal_to_modulus = false;

// static
template <typename T>
bool RapidJsonValueConverter<
    T, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>, T>>>::
    s_is_in_montgomery = false;

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_
