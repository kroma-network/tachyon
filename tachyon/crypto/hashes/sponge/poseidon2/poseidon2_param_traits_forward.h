#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PARAM_TRAITS_FORWARD_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PARAM_TRAITS_FORWARD_H_

#include <stddef.h>
#include <stdint.h>

namespace tachyon::crypto {

template <typename F, size_t Rate, uint32_t Alpha>
struct Poseidon2ParamsTraits;

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PARAM_TRAITS_FORWARD_H_
