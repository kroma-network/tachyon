#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_

#include <cmath>

#include "tachyon/base/bits.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/math/base/field.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/prime_field_forward.h"
#include "tachyon/math/finite_fields/prime_field_traits.h"
#include "tachyon/math/finite_fields/prime_field_util.h"

namespace tachyon::math {

template <typename F>
class PrimeFieldBase : public Field<F> {
 public:
  using Config = typename PrimeFieldTraits<F>::Config;

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

  constexpr static uint64_t ExtensionDegree() { return 1; }

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
      uint32_t q_adicity =
          ComputeAdicity(Config::kSmallSubgroupBase, gmp::FromUnsignedInt(n));
      uint64_t q_part = static_cast<uint64_t>(
          std::pow(Config::kSmallSubgroupBase, q_adicity));

      uint32_t two_adicity = ComputeAdicity(2, gmp::FromUnsignedInt(n));
      uint64_t two_part = static_cast<uint64_t>(std::pow(2, two_adicity));

      if (n != two_part * q_part || two_adicity > Config::kTwoAdicity ||
          q_adicity > Config::kSmallSubgroupAdicity) {
        return false;
      }

      omega = F::FromMontgomery(Config::kLargeSubgroupRootOfUnity);
      for (size_t i = q_adicity; i < Config::kSmallSubgroupAdicity; ++i) {
        omega = omega.Pow(BigInt<1>(Config::kSmallSubgroupBase));
      }

      for (size_t i = two_adicity; i < Config::kTwoAdicity; ++i) {
        omega.SquareInPlace();
      }
    } else {
      uint64_t log_size_of_group =
          static_cast<uint64_t>(base::bits::Log2Ceiling(n));
      uint64_t size = uint64_t{1} << log_size_of_group;

      if (n != size || log_size_of_group > Config::kTwoAdicity) {
        return false;
      }

      omega = F::FromMontgomery(Config::kTwoAdicRootOfUnity);
      for (size_t i = log_size_of_group; i < Config::kTwoAdicity; ++i) {
        omega.SquareInPlace();
      }
    }
    *ret = omega;
    return true;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_
