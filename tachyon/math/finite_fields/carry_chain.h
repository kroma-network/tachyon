#ifndef TACHYON_MATH_FINITE_FIELDS_CARRY_CHAIN_H_
#define TACHYON_MATH_FINITE_FIELDS_CARRY_CHAIN_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>

#include "tachyon/math/finite_fields/prime_field_ops_internal.h"

namespace tachyon::math {
namespace u32 {

template <size_t OpsCount = std::numeric_limits<size_t>::max(),
          bool CarryIn = false, bool CarryOut = false>
struct CarryChain {
  size_t index = 0;

  AddResult<uint32_t> Add(uint32_t x, uint32_t y, uint32_t carry = 0) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return internal::u32::Add(x, y);
    } else if (index == 1 && !CarryIn) {
      return internal::u32::AddCc(x, y);
    } else if (index < OpsCount || CarryOut) {
      return internal::u32::AddcCc(x, y, carry);
    } else {
      return internal::u32::Addc(x, y, carry);
    }
  }

  SubResult<uint32_t> Sub(uint32_t x, uint32_t y, uint32_t carry = 0) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return internal::u32::Sub(x, y);
    } else if (index == 1 && !CarryIn) {
      return internal::u32::SubCc(x, y);
    } else if (index < OpsCount || CarryOut) {
      return internal::u32::SubcCc(x, y, carry);
    } else {
      return internal::u32::Subc(x, y, carry);
    }
  }

  AddResult<uint32_t> MadLo(uint32_t x, uint32_t y, uint32_t z,
                            uint32_t carry = 0) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return internal::u32::MadLo(x, y, z);
    } else if (index == 1 && !CarryIn) {
      return internal::u32::MadLoCc(x, y, z);
    } else if (index < OpsCount || CarryOut) {
      return internal::u32::MadcLoCc(x, y, z, carry);
    } else {
      return internal::u32::MadcLo(x, y, z, carry);
    }
  }

  AddResult<uint32_t> MadHi(uint32_t x, uint32_t y, uint32_t z,
                            uint32_t carry = 0) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return internal::u32::MadHi(x, y, z);
    } else if (index == 1 && !CarryIn) {
      return internal::u32::MadHiCc(x, y, z);
    } else if (index < OpsCount || CarryOut) {
      return internal::u32::MadcHiCc(x, y, z, carry);
    } else {
      return internal::u32::MadcHi(x, y, z, carry);
    }
  }
};

}  // namespace u32

namespace u64 {

template <size_t OpsCount = std::numeric_limits<size_t>::max(),
          bool CarryIn = false, bool CarryOut = false>
struct CarryChain {
  size_t index = 0;

  AddResult<uint64_t> Add(uint64_t x, uint64_t y, uint64_t carry = 0) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return internal::u64::Add(x, y);
    } else if (index == 1 && !CarryIn) {
      return internal::u64::AddCc(x, y);
    } else if (index < OpsCount || CarryOut) {
      return internal::u64::AddcCc(x, y, carry);
    } else {
      return internal::u64::Addc(x, y, carry);
    }
  }

  SubResult<uint64_t> Sub(uint64_t x, uint64_t y, uint64_t carry = 0) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return internal::u64::Sub(x, y);
    } else if (index == 1 && !CarryIn) {
      return internal::u64::SubCc(x, y);
    } else if (index < OpsCount || CarryOut) {
      return internal::u64::SubcCc(x, y, carry);
    } else {
      return internal::u64::Subc(x, y, carry);
    }
  }

  AddResult<uint64_t> MadLo(uint64_t x, uint64_t y, uint64_t z,
                            uint64_t carry = 0) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return internal::u64::MadLo(x, y, z);
    } else if (index == 1 && !CarryIn) {
      return internal::u64::MadLoCc(x, y, z);
    } else if (index < OpsCount || CarryOut) {
      return internal::u64::MadcLoCc(x, y, z, carry);
    } else {
      return internal::u64::MadcLo(x, y, z, carry);
    }
  }

  AddResult<uint64_t> MadHi(uint64_t x, uint64_t y, uint64_t z,
                            uint64_t carry = 0) {
    ++index;
    if (index == 1 && OpsCount == 1 && !CarryIn && !CarryOut) {
      return internal::u64::MadHi(x, y, z);
    } else if (index == 1 && !CarryIn) {
      return internal::u64::MadHiCc(x, y, z);
    } else if (index < OpsCount || CarryOut) {
      return internal::u64::MadcHiCc(x, y, z, carry);
    } else {
      return internal::u64::MadcHi(x, y, z, carry);
    }
  }
};

}  // namespace u64
}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_CARRY_CHAIN_H_
