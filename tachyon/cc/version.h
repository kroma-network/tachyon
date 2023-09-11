#ifndef TACHYON_CC_VERSION_H_
#define TACHYON_CC_VERSION_H_

#include <stdint.h>

#include <string_view>

#include "tachyon/cc/export.h"
#include "tachyon/cc/version_generated.h"

namespace tachyon::cc {

TACHYON_CC_EXPORT uint32_t GetRuntimeVersion();

TACHYON_CC_EXPORT std::string_view GetRuntimeVersionStr();

TACHYON_CC_EXPORT std::string_view GetRuntimeFullVersionStr();

}  // namespace tachyon::cc

#endif  // TACHYON_CC_VERSION_H_
