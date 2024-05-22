#include "circomlib/base/modulus.h"

namespace tachyon::circom {

std::string Modulus::ToString() const {
#define MODULUS_TO_STRING(n) ToBigInt<n>().ToString()
  switch (bytes.size() / 8) {
    case 1:
      return MODULUS_TO_STRING(1);
    case 2:
      return MODULUS_TO_STRING(2);
    case 3:
      return MODULUS_TO_STRING(3);
    case 4:
      return MODULUS_TO_STRING(4);
    case 5:
      return MODULUS_TO_STRING(5);
    case 6:
      return MODULUS_TO_STRING(6);
    case 7:
      return MODULUS_TO_STRING(7);
    case 8:
      return MODULUS_TO_STRING(8);
    case 9:
      return MODULUS_TO_STRING(9);
  }
#undef MODULUS_TO_STRING
  NOTREACHED();
  return "";
}

}  // namespace tachyon::circom
