#ifndef TACHYON_BASE_OPENMP_UTIL_H_
#define TACHYON_BASE_OPENMP_UTIL_H_

#include <algorithm>

#if defined(TACHYON_HAS_OPENMP)
#include <omp.h>
#endif  // defined(TACHYON_HAS_OPENMP)

#if defined(TACHYON_HAS_OPENMP)
#define OPENMP_PARALLEL_FOR(expr) _Pragma("omp parallel for") for (expr)
#else
#define OPENMP_PARALLEL_FOR(expr) for (expr)
#endif  // defined(TACHYON_HAS_OPENMP)

namespace tachyon::base {

constexpr static size_t kDefaultNumElements = 1024;

template <typename ContainerType>
static inline size_t GetNumElementsPerThread(const ContainerType& container) {
#if defined(TACHYON_HAS_OPENMP)
  size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
  size_t thread_nums = 1;
#endif
  size_t size = container.size();
  return std::max(size / thread_nums, kDefaultNumElements);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_OPENMP_UTIL_H_
