#ifndef TACHYON_BASE_OPENMP_UTIL_H_
#define TACHYON_BASE_OPENMP_UTIL_H_

#include <algorithm>
#include <optional>

#if defined(TACHYON_HAS_OPENMP)
#include <omp.h>
#endif  // defined(TACHYON_HAS_OPENMP)

#if defined(TACHYON_HAS_OPENMP)
#define OPENMP_PARALLEL_FOR(expr) _Pragma("omp parallel for") for (expr)
#else
#define OPENMP_PARALLEL_FOR(expr) for (expr)
#endif  // defined(TACHYON_HAS_OPENMP)

namespace tachyon::base {

// NOTE(chokobole): This function might return 0. You should handle this case
// carefully. See other examples where it is used.
template <typename Container>
size_t GetNumElementsPerThread(const Container& container,
                               std::optional<size_t> threshold = std::nullopt) {
#if defined(TACHYON_HAS_OPENMP)
  size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
  size_t thread_nums = 1;
#endif
  size_t size = std::size(container);
  return (!threshold.has_value() || size > threshold.value())
             ? (size + thread_nums - 1) / thread_nums
             : size;
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_OPENMP_UTIL_H_
