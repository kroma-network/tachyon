#ifndef TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_
#define TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_

#include <type_traits>

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/base/compiler_specific.h"

namespace tachyon::device::gpu {

enum class StateSpace {
  kNone,
  kGlobal,
};

enum class CacheOperator {
  kNone,
  // Shared by store and load instruction
  // Cache at global level(cache in L2 and below, not L1).
  kGlobal,
  // Cache streaming, likely to be accessed once.
  kStreaming,
  // Load instruction only
  // Cache at all levels, likely to be accessed again.
  kAll,
  // Last use.
  kLastUse,
  // Cache as volatile (consider cached system memory lines stale, fetch again).
  kVolatile,
  // Store instruction only
  // Cache write-back all coherent levels.
  kWriteBack,
  // Cache write-through.
  kWriteThrough,
};

template <typename T, StateSpace Space, CacheOperator Operator>
__device__ ALWAYS_INLINE constexpr T LoadSingle(const T* ptr) {
  if constexpr (Space == StateSpace::kGlobal) {
    if constexpr (Operator == CacheOperator::kNone) {
      return __ldg(ptr);
    } else if constexpr (Operator == CacheOperator::kAll) {
      return __ldca(ptr);
    } else if constexpr (Operator == CacheOperator::kGlobal) {
      return __ldcg(ptr);
    } else if constexpr (Operator == CacheOperator::kStreaming) {
      return __ldcs(ptr);
    } else if constexpr (Operator == CacheOperator::kLastUse) {
      return __ldlu(ptr);
    } else if constexpr (Operator == CacheOperator::kVolatile) {
      return __ldcv(ptr);
    }
  }
  return *ptr;
}

template <class T, typename U, CacheOperator Operator, size_t Stride>
__device__ ALWAYS_INLINE constexpr T Load(const T* address, size_t offset) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % sizeof(U) == 0);
  constexpr size_t count = sizeof(T) / sizeof(U);
  T result = {};
  auto pa = reinterpret_cast<const U*>(address) + offset;
  auto pr = reinterpret_cast<U*>(&result);
  for (size_t i = 0; i < count; ++i) {
    pr[i] = LoadSingle<U, StateSpace::kGlobal, Operator>(&pa[i * Stride]);
  }
  return result;
}

template <class T, CacheOperator Operator = CacheOperator::kNone,
          size_t Stride = 1,
          typename U = std::enable_if_t<sizeof(T) % sizeof(uint4) == 0, uint4>>
__device__ ALWAYS_INLINE constexpr T Load(const T* address,
                                          const size_t offset = 0,
                                          [[maybe_unused]] uint4 _dummy = {}) {
  return Load<T, U, Operator, Stride>(address, offset);
}

template <typename T, CacheOperator Operator>
__device__ ALWAYS_INLINE constexpr void StoreSingle(T* ptr, T value) {
  if constexpr (Operator == CacheOperator::kWriteBack) {
    __stwb(ptr, value);
  } else if constexpr (Operator == CacheOperator::kGlobal) {
    __stcg(ptr, value);
  } else if constexpr (Operator == CacheOperator::kStreaming) {
    __stcs(ptr, value);
  } else if constexpr (Operator == CacheOperator::kWriteThrough) {
    __stwt(ptr, value);
  } else {
    *ptr = value;
  }
}

// T=uint4, U=uint4, Operator=tachyon::device::gpu::CacheOperator::kStreaming,
// Stride=1UL]"
template <class T, typename U, CacheOperator Operator, size_t Stride>
__device__ ALWAYS_INLINE constexpr void Store(T* address, T value,
                                              size_t offset) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % sizeof(U) == 0);
  constexpr size_t count = sizeof(T) / sizeof(U);
  auto pa = reinterpret_cast<U*>(address) + offset;
  auto pv = reinterpret_cast<const U*>(&value);
  for (size_t i = 0; i < count; ++i) {
    StoreSingle<U, Operator>(&pa[i * Stride], pv[i]);
  }
}

template <class T, CacheOperator Operator = CacheOperator::kNone,
          size_t Stride = 1,
          typename U = std::enable_if_t<sizeof(T) % sizeof(uint4) == 0, uint4>>
__device__ ALWAYS_INLINE constexpr void Store(
    T* address, const T& value, const size_t offset = 0,
    [[maybe_unused]] uint4 _dummy = {}) {
  Store<T, U, Operator, Stride>(address, value, offset);
}

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_
