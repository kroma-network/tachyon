#include "tachyon/version.h"

namespace tachyon {

uint32_t GetRuntimeVersion() { return TACHYON_VERSION; }

std::string_view GetRuntimeVersionStr() { return TACHYON_VERSION_STR; }

std::string_view GetRuntimeFullVersionStr() { return TACHYON_VERSION_FULL_STR; }

}  // namespace tachyon