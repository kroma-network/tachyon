/**
 * @file profiler.h
 * @brief Profiler interface.
 *
 * This header file provides an interface for profiler.
 *
 * @example profiler.cc
 */
#ifndef TACHYON_C_BASE_PROFILER_H_
#define TACHYON_C_BASE_PROFILER_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"

/**
 * @struct tachyon_profiler
 * @brief Represents a profiler.
 */
struct tachyon_profiler {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a profiler.
 * @param path The path to the file to record profiling result.
 * @param path_len The length of the path.
 * @return Pointer to the newly created profiler.
 */
TACHYON_C_EXPORT tachyon_profiler* tachyon_profiler_create(const char* path,
                                                           size_t path_len);

/**
 * @brief Destroys a profiler.
 *
 * Frees the resources associated with the profiler. After
 * calling this function, the profiler pointer should not be used anymore.
 *
 * @param profiler A pointer to the profiler to destroy.
 */
TACHYON_C_EXPORT void tachyon_profiler_destroy(tachyon_profiler* profiler);

/**
 * @brief Initializes a profiler.
 * @param profiler A pointer to the profiler.
 */
TACHYON_C_EXPORT void tachyon_profiler_init(tachyon_profiler* profiler);

/**
 * @brief Starts a profiler.
 * @param profiler A pointer to the profiler.
 */
TACHYON_C_EXPORT void tachyon_profiler_start(tachyon_profiler* profiler);

/**
 * @brief Stops a profiler.
 * @param profiler A pointer to the profiler.
 */
TACHYON_C_EXPORT void tachyon_profiler_stop(tachyon_profiler* profiler);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_BASE_PROFILER_H_
