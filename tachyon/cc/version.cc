#include "tachyon/cc/version.h"

namespace tachyon::cc {

uint32_t GetRuntimeVersion() { return TACHYON_CC_VERSION; }

std::string_view GetRuntimeVersionStr() { return TACHYON_CC_VERSION_STR; }

std::string_view GetRuntimeFullVersionStr() {
  return TACHYON_CC_VERSION_FULL_STR;
}

}  // namespace tachyon::cc
