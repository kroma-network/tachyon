#include "tachyon/base/endian.h"

#include "tachyon/base/logging.h"

namespace tachyon::base {

std::string_view EndianToString(Endian endian) {
  switch (endian) {
    case Endian::kNative:
      return "Native";
    case Endian::kBig:
      return "Big";
    case Endian::kLittle:
      return "Little";
  }
  NOTREACHED();
  return "";
}

std::ostream& operator<<(std::ostream& os, Endian endian) {
  return os << EndianToString(endian);
}

}  // namespace tachyon::base
