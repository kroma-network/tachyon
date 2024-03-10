#include "circomlib/r1cs/r1cs.h"

namespace tachyon::circom::v1 {

std::string_view R1CSSectionTypeToString(R1CSSectionType type) {
  switch (type) {
    case R1CSSectionType::kHeader:
      return "Header";
    case R1CSSectionType::kConstraints:
      return "Constraints";
    case R1CSSectionType::kWire2LabelIdMap:
      return "Wire2LabelIdMap";
    case R1CSSectionType::kCustomGatesList:
      return "CustomGatesList";
    case R1CSSectionType::kCustomGatesApplication:
      return "CustomGatesApplication";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::circom::v1
