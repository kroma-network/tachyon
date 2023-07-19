#ifndef TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_INTERNAL_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_INTERNAL_CU_H_

#include <stdint.h>

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

#include "tachyon/base/compiler_specific.h"

namespace tachyon {
namespace math {
namespace ptx {
namespace u32 {

__device__ ALWAYS_INLINE uint32_t Add(uint32_t x, uint32_t y) {
  uint32_t result;
  asm("add.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t AddCc(uint32_t x, uint32_t y) {
  uint32_t result;
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t Addc(uint32_t x, uint32_t y) {
  uint32_t result;
  asm volatile("addc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t AddcCc(uint32_t x, uint32_t y) {
  uint32_t result;
  asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t Sub(uint32_t x, uint32_t y) {
  uint32_t result;
  asm("sub.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t SubCc(uint32_t x, uint32_t y) {
  uint32_t result;
  asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t Subc(uint32_t x, uint32_t y) {
  uint32_t result;
  asm volatile("subc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t SubcCc(uint32_t x, uint32_t y) {
  uint32_t result;
  asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MulLo(uint32_t x, uint32_t y) {
  uint32_t result;
  asm("mul.lo.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MulHi(uint32_t x, uint32_t y) {
  uint32_t result;
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MadLo(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MadHi(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MadLoCc(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result;
  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MadHiCc(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result;
  asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MadcLo(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result;
  asm volatile("madc.lo.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MadcHi(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result;
  asm volatile("madc.hi.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MadcLoCc(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result;
  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ ALWAYS_INLINE uint32_t MadcHiCc(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result;
  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MovB64(uint32_t lo, uint32_t hi) {
  uint64_t result;
  asm("mov.b64 %0, {%1,%2};" : "=l"(result) : "r"(lo), "r"(hi));
  return result;
}

}  // namespace u32

namespace u64 {

__device__ ALWAYS_INLINE uint64_t Add(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("add.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t AddCc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t Addc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("addc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t AddcCc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t Sub(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("sub.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t SubCc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t Subc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("subc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t SubcCc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MulLo(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("mul.lo.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MulHi(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("mul.hi.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MadLo(const uint64_t x, const uint64_t y,
                                        const uint64_t z) {
  uint64_t result;
  asm("mad.lo.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MadHi(const uint64_t x, const uint64_t y,
                                        const uint64_t z) {
  uint64_t result;
  asm("mad.hi.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MadLoCc(const uint64_t x, const uint64_t y,
                                          const uint64_t z) {
  uint64_t result;
  asm volatile("mad.lo.cc.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MadHiCc(const uint64_t x, const uint64_t y,
                                          const uint64_t z) {
  uint64_t result;
  asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MadcLo(const uint64_t x, const uint64_t y,
                                         const uint64_t z) {
  uint64_t result;
  asm volatile("madc.lo.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MadcHi(const uint64_t x, const uint64_t y,
                                         const uint64_t z) {
  uint64_t result;
  asm volatile("madc.hi.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MadcLoCc(const uint64_t x, const uint64_t y,
                                           const uint64_t z) {
  uint64_t result;
  asm volatile("madc.lo.cc.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ ALWAYS_INLINE uint64_t MadcHiCc(const uint64_t x, const uint64_t y,
                                           const uint64_t z) {
  uint64_t result;
  asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

}  // namespace u64

__device__ ALWAYS_INLINE void BarArrive(const unsigned name,
                                        const unsigned count) {
  asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(count) : "memory");
}

__device__ ALWAYS_INLINE void BarSync(const unsigned name,
                                      const unsigned count) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(count) : "memory");
}

}  // namespace ptx
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_INTERNAL_CU_H_
