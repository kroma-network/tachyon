#ifndef TACHYON_C_VERSION_H_
#define TACHYON_C_VERSION_H_

#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/version_generated.h"

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT uint32_t tachyon_get_runtime_version();

TACHYON_C_EXPORT const char* tachyon_get_runtime_version_str();

TACHYON_C_EXPORT const char* tachyon_get_runtime_full_version_str();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_VERSION_H_
