#ifndef TACHYON_BASE_ENDIAN_H_
#define TACHYON_BASE_ENDIAN_H_

#include <ostream>

#include "tachyon/export.h"

namespace tachyon::base {

enum class Endian {
  kNative,
  kBig,
  kLittle,
};

TACHYON_EXPORT std::string_view EndianToString(Endian endian);

TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, Endian endian);

}  // namespace tachyon::base

#endif  // TACHYON_BASE_ENDIAN_H_
