#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_OPS_INTERNAL_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_OPS_INTERNAL_H_

#include <stdint.h>

#include "tachyon/base/compiler_specific.h"
#include "tachyon/math/base/arithmetics.h"

namespace tachyon::math::internal {
namespace u32 {

ALWAYS_INLINE AddResult<uint32_t> Add(uint32_t x, uint32_t y) {
  AddResult<uint32_t> result = AddWithCarry(x, y);
  result.carry = 0;
  return result;
}

ALWAYS_INLINE AddResult<uint32_t> AddCc(uint32_t x, uint32_t y) {
  return AddWithCarry(x, y);
}

ALWAYS_INLINE AddResult<uint32_t> Addc(uint32_t x, uint32_t y, uint32_t carry) {
  AddResult<uint32_t> result = AddWithCarry(x, y, carry);
  result.carry = 0;
  return result;
}

ALWAYS_INLINE AddResult<uint32_t> AddcCc(uint32_t x, uint32_t y,
                                         uint32_t carry) {
  return AddWithCarry(x, y, carry);
}

ALWAYS_INLINE SubResult<uint32_t> Sub(uint32_t x, uint32_t y) {
  SubResult<uint32_t> result = SubWithBorrow(x, y);
  result.borrow = 0;
  return result;
}

ALWAYS_INLINE SubResult<uint32_t> SubCc(uint32_t x, uint32_t y) {
  return SubWithBorrow(x, y);
}

ALWAYS_INLINE SubResult<uint32_t> Subc(uint32_t x, uint32_t y,
                                       uint32_t borrow) {
  SubResult<uint32_t> result = SubWithBorrow(x, y, borrow);
  result.borrow = 0;
  return result;
}

ALWAYS_INLINE SubResult<uint32_t> SubcCc(uint32_t x, uint32_t y,
                                         uint32_t borrow) {
  return SubWithBorrow(x, y, borrow);
}

ALWAYS_INLINE constexpr uint32_t MulLo(uint32_t x, uint32_t y) {
  return MulAddWithCarry(0, x, y).lo;
}

ALWAYS_INLINE constexpr uint32_t MulHi(uint32_t x, uint32_t y) {
  return MulAddWithCarry(0, x, y).hi;
}

ALWAYS_INLINE AddResult<uint32_t> MadLo(uint32_t x, uint32_t y, uint32_t z) {
  return Add(MulLo(x, y), z);
}

ALWAYS_INLINE AddResult<uint32_t> MadHi(uint32_t x, uint32_t y, uint32_t z) {
  return Add(MulHi(x, y), z);
}

ALWAYS_INLINE AddResult<uint32_t> MadLoCc(uint32_t x, uint32_t y, uint32_t z) {
  return AddCc(MulLo(x, y), z);
}

ALWAYS_INLINE AddResult<uint32_t> MadHiCc(uint32_t x, uint32_t y, uint32_t z) {
  return AddCc(MulHi(x, y), z);
}

ALWAYS_INLINE AddResult<uint32_t> MadcLo(uint32_t x, uint32_t y, uint32_t z,
                                         uint32_t carry) {
  return Addc(MulLo(x, y), z, carry);
}

ALWAYS_INLINE AddResult<uint32_t> MadcHi(uint32_t x, uint32_t y, uint32_t z,
                                         uint32_t carry) {
  return Addc(MulHi(x, y), z, carry);
}

ALWAYS_INLINE AddResult<uint32_t> MadcLoCc(uint32_t x, uint32_t y, uint32_t z,
                                           uint32_t carry) {
  return AddcCc(MulLo(x, y), z, carry);
}

ALWAYS_INLINE AddResult<uint32_t> MadcHiCc(uint32_t x, uint32_t y, uint32_t z,
                                           uint32_t carry) {
  return AddcCc(MulHi(x, y), z, carry);
}

}  // namespace u32

namespace u64 {

ALWAYS_INLINE AddResult<uint64_t> Add(uint64_t x, uint64_t y) {
  AddResult<uint64_t> result = AddWithCarry(x, y);
  result.carry = 0;
  return result;
}

ALWAYS_INLINE AddResult<uint64_t> AddCc(uint64_t x, uint64_t y) {
  return AddWithCarry(x, y);
}

ALWAYS_INLINE AddResult<uint64_t> Addc(uint64_t x, uint64_t y, uint64_t carry) {
  AddResult<uint64_t> result = AddWithCarry(x, y, carry);
  result.carry = 0;
  return result;
}

ALWAYS_INLINE AddResult<uint64_t> AddcCc(uint64_t x, uint64_t y,
                                         uint64_t carry) {
  return AddWithCarry(x, y, carry);
}

ALWAYS_INLINE SubResult<uint64_t> Sub(uint64_t x, uint64_t y) {
  SubResult<uint64_t> result = SubWithBorrow(x, y);
  result.borrow = 0;
  return result;
}

ALWAYS_INLINE SubResult<uint64_t> SubCc(uint64_t x, uint64_t y) {
  return SubWithBorrow(x, y);
}

ALWAYS_INLINE SubResult<uint64_t> Subc(uint64_t x, uint64_t y,
                                       uint64_t borrow) {
  SubResult<uint64_t> result = SubWithBorrow(x, y, borrow);
  result.borrow = 0;
  return result;
}

ALWAYS_INLINE SubResult<uint64_t> SubcCc(uint64_t x, uint64_t y,
                                         uint64_t borrow) {
  return SubWithBorrow(x, y, borrow);
}

ALWAYS_INLINE uint64_t MulLo(uint64_t x, uint64_t y) {
  return MulAddWithCarry(0, x, y).lo;
}

ALWAYS_INLINE uint64_t MulHi(uint64_t x, uint64_t y) {
  return MulAddWithCarry(0, x, y).hi;
}

ALWAYS_INLINE AddResult<uint64_t> MadLo(uint64_t x, uint64_t y, uint64_t z) {
  return Add(MulLo(x, y), z);
}

ALWAYS_INLINE AddResult<uint64_t> MadHi(uint64_t x, uint64_t y, uint64_t z) {
  return Add(MulHi(x, y), z);
}

ALWAYS_INLINE AddResult<uint64_t> MadLoCc(uint64_t x, uint64_t y, uint64_t z) {
  return AddCc(MulLo(x, y), z);
}

ALWAYS_INLINE AddResult<uint64_t> MadHiCc(uint64_t x, uint64_t y, uint64_t z) {
  return AddCc(MulHi(x, y), z);
}

ALWAYS_INLINE AddResult<uint64_t> MadcLo(uint64_t x, uint64_t y, uint64_t z,
                                         uint64_t carry) {
  return Addc(MulLo(x, y), z, carry);
}

ALWAYS_INLINE AddResult<uint64_t> MadcHi(uint64_t x, uint64_t y, uint64_t z,
                                         uint64_t carry) {
  return Addc(MulHi(x, y), z, carry);
}

ALWAYS_INLINE AddResult<uint64_t> MadcLoCc(uint64_t x, uint64_t y, uint64_t z,
                                           uint64_t carry) {
  return AddcCc(MulLo(x, y), z, carry);
}

ALWAYS_INLINE AddResult<uint64_t> MadcHiCc(uint64_t x, uint64_t y, uint64_t z,
                                           uint64_t carry) {
  return AddcCc(MulHi(x, y), z, carry);
}

}  // namespace u64
}  // namespace tachyon::math::internal

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_OPS_INTERNAL_H_
