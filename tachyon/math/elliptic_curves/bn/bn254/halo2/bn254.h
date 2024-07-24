#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_HALO2_BN254_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_HALO2_BN254_H_

#include "tachyon/export.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon::math::halo2 {

TACHYON_EXPORT void OverrideSubgroupGenerator();

struct TACHYON_EXPORT ScopedSubgroupGeneratorOverrider {
  BigInt<4> subgroup_generator;
  BigInt<4> two_adic_root_of_unity;
  BigInt<4> large_subgroup_root_of_unity;

  ScopedSubgroupGeneratorOverrider();
  ~ScopedSubgroupGeneratorOverrider();
};

}  // namespace tachyon::math::halo2

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_HALO2_BN254_H_
