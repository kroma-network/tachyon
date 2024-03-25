#ifndef TACHYON_C_VERSION_H_
#define TACHYON_C_VERSION_H_

#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/version_generated.h"

/**
 * @file version.h
 * @brief Version information for tachyon.
 *
 * This header file contains functions to get the runtime version information of
 * tachyon.
 *
 * @example version.cc
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the runtime version number of tachyon.
 * @return Runtime version number as a uint32_t. e.g., 10203.
 */
TACHYON_C_EXPORT uint32_t tachyon_get_runtime_version();

/**
 * @brief Returns the runtime version string of tachyon.
 * @return Runtime version string as a std::string_view. e.g., "1.2.3".
 */
TACHYON_C_EXPORT const char* tachyon_get_runtime_version_str();

/**
 * @brief Returns the runtime full version string of tachyon.
 * @return Runtime full version string as a std::string_view. e.g.,
 * "1.2.3-<commit sha256>".
 */
TACHYON_C_EXPORT const char* tachyon_get_runtime_full_version_str();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_VERSION_H_
