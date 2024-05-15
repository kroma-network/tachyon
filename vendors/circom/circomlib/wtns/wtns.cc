#include "circomlib/wtns/wtns.h"

namespace tachyon::circom::v2 {

std::string_view WtnsSectionTypeToString(WtnsSectionType type) {
  switch (type) {
    case WtnsSectionType::kHeader:
      return "Header";
    case WtnsSectionType::kData:
      return "Data";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::circom::v2
