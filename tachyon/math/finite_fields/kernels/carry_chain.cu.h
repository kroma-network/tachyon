#ifndef TACHYON_MATH_FINITE_FIELDS_KERNELS_CARRY_CHAIN_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_KERNELS_CARRY_CHAIN_CU_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>

#include "tachyon/math/finite_fields/kernels/prime_field_ops_internal.cu.h"

namespace tachyon {
namespace math {

template <size_t OpsCount = std::numeric_limits<size_t>::max(),
          bool CarryIn = false, bool CarryOut = false>
struct CarryChain {
  size_t index = 0;

  __device__ uint64_t Add(uint64_t x, uint64_t y) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return ptx::u64::Add(x, y);
    } else if (index == 1 && !CarryIn) {
      return ptx::u64::AddCc(x, y);
    } else if (index < OpsCount || CarryOut) {
      return ptx::u64::AddcCc(x, y);
    } else {
      return ptx::u64::Addc(x, y);
    }
  }

  __device__ uint64_t Sub(uint64_t x, uint64_t y) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return ptx::u64::Sub(x, y);
    } else if (index == 1 && !CarryIn) {
      return ptx::u64::SubCc(x, y);
    } else if (index < OpsCount || CarryOut) {
      return ptx::u64::SubcCc(x, y);
    } else {
      return ptx::u64::Subc(x, y);
    }
  }

  __device__ uint64_t MadLo(uint64_t x, uint64_t y, uint64_t z) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return ptx::u64::MadLo(x, y, z);
    } else if (index == 1 && !CarryIn) {
      return ptx::u64::MadLoCc(x, y, z);
    } else if (index < OpsCount || CarryOut) {
      return ptx::u64::MadcLoCc(x, y, z);
    } else {
      return ptx::u64::MadcLo(x, y, z);
    }
  }

  __device__ uint64_t MadHi(uint64_t x, uint64_t y, uint64_t z) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return ptx::u64::MadHi(x, y, z);
    } else if (index == 1 && !CarryIn) {
      return ptx::u64::MadHiCc(x, y, z);
    } else if (index < OpsCount || CarryOut) {
      return ptx::u64::MadcHiCc(x, y, z);
    } else {
      return ptx::u64::MadcHi(x, y, z);
    }
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_KERNELS_CARRY_CHAIN_CU_H_
