#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_VENDOR_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_VENDOR_H_

#include <string>

#include "tachyon/export.h"

namespace tachyon::crypto {

enum class Poseidon2Vendor {
  kHorizen,
  kPlonky3,
};

TACHYON_EXPORT std::string Poseidon2VendorToString(Poseidon2Vendor vendor);

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_VENDOR_H_
