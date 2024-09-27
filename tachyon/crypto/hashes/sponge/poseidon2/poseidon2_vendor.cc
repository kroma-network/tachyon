#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_vendor.h"

#include <string>

#include "tachyon/base/logging.h"

namespace tachyon::crypto {

std::string Poseidon2VendorToString(Poseidon2Vendor vendor) {
  switch (vendor) {
    case Poseidon2Vendor::kHorizen:
      return "horizen";
    case Poseidon2Vendor::kPlonky3:
      return "plonky3";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::crypto
