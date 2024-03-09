#include "circomlib/r1cs/r1cs.h"

namespace tachyon::circom {

std::string PrimeField::ToString() const {
#define PRIME_FIELD_TO_STRING(n) ToBigInt<n>().ToString()
  switch (bytes.size() / 8) {
    case 1:
      return PRIME_FIELD_TO_STRING(1);
    case 2:
      return PRIME_FIELD_TO_STRING(2);
    case 3:
      return PRIME_FIELD_TO_STRING(3);
    case 4:
      return PRIME_FIELD_TO_STRING(4);
    case 5:
      return PRIME_FIELD_TO_STRING(5);
    case 6:
      return PRIME_FIELD_TO_STRING(6);
    case 7:
      return PRIME_FIELD_TO_STRING(7);
    case 8:
      return PRIME_FIELD_TO_STRING(8);
    case 9:
      return PRIME_FIELD_TO_STRING(9);
  }
#undef PRIME_FIELD_TO_STRING
  NOTREACHED();
  return "";
}

namespace v1 {

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

}  // namespace v1
}  // namespace tachyon::circom
