#ifndef TACHYON_VERSION_H_
#define TACHYON_VERSION_H_

#include <stdint.h>

#include <string_view>

#include "tachyon/export.h"
#include "tachyon/version_generated.h"

namespace tachyon {

TACHYON_EXPORT uint32_t GetRuntimeVersion();

TACHYON_EXPORT std::string_view GetRuntimeVersionStr();

TACHYON_EXPORT std::string_view GetRuntimeFullVersionStr();

}  // namespace tachyon

#endif  // TACHYON_VERSION_H_