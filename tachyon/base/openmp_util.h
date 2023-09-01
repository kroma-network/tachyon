#ifndef TACHYON_BASE_OPENMP_UTIL_H_
#define TACHYON_BASE_OPENMP_UTIL_H_

#if defined(TACHYON_HAS_OPENMP)
#include <omp.h>
#endif  // defined(TACHYON_HAS_OPENMP)

#if defined(TACHYON_HAS_OPENMP)
#define OPENMP_PARALLEL_FOR(expr) _Pragma("omp parallel for") for (expr)
#else
#define OPENMP_PARALLEL_FOR(expr) for (expr)
#endif  // defined(TACHYON_HAS_OPENMP)

#endif  // TACHYON_BASE_OPENMP_UTIL_H_
