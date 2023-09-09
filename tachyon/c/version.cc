#include "tachyon/c/version.h"

uint32_t tachyon_get_runtime_version() { return TACHYON_C_VERSION; }

const char* tachyon_get_runtime_version_str() { return TACHYON_C_VERSION_STR; }

const char* tachyon_get_runtime_full_version_str() {
  return TACHYON_C_VERSION_FULL_STR;
}
