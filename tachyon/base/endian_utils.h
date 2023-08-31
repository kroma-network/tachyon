#ifndef TACHYON_BASE_ENDIAN_UTILS_H_
#define TACHYON_BASE_ENDIAN_UTILS_H_

#include "tachyon/build/build_config.h"

#if ARCH_CPU_BIG_ENDIAN
#define FOR_FROM_BIGGEST(idx, start, end) \
  for (size_t idx = start; idx < end; ++idx)
#else  // ARCH_CPU_LITTLE_ENDIAN
#define FOR_FROM_BIGGEST(idx, start, end) \
  for (size_t idx = end - 1; idx != static_cast<size_t>(start - 1); --idx)
#endif

#if ARCH_CPU_BIG_ENDIAN
#define FOR_FROM_SMALLEST(idx, start, end) \
  for (size_t idx = end - 1; idx != static_cast<size_t>(start - 1); --idx)
#else  // ARCH_CPU_LITTLE_ENDIAN
#define FOR_FROM_SMALLEST(idx, start, end) \
  for (size_t idx = start; idx < end; ++idx)
#endif

#if ARCH_CPU_BIG_ENDIAN
#define FOR_FROM_SECOND_SMALLEST(idx, start, end) \
  for (size_t idx = end - 2; idx != static_cast<size_t>(start - 1); --idx)
#else  // ARCH_CPU_LITTLE_ENDIAN
#define FOR_FROM_SECOND_SMALLEST(idx, start, end) \
  for (size_t idx = start + 1; idx < end; ++idx)
#endif

#if ARCH_CPU_BIG_ENDIAN
#define FOR_BUT_SMALLEST(idx, end) for (size_t idx = 0; idx != end - 1; ++idx)
#else  // ARCH_CPU_LITTLE_ENDIAN
#define FOR_BUT_SMALLEST(idx, end) for (size_t idx = 1; idx < end; ++idx)
#endif

#if ARCH_CPU_BIG_ENDIAN
#define SMALLEST_INDEX(end) end - 1
#else  // ARCH_CPU_LITTLE_ENDIAN
#define SMALLEST_INDEX(end) 0
#endif

#if ARCH_CPU_BIG_ENDIAN
#define BIGGEST_INDEX(end) 0
#else  // ARCH_CPU_LITTLE_ENDIAN
#define BIGGEST_INDEX(end) end - 1
#endif

#endif  // TACHYON_BASE_ENDIAN_UTILS_H_
