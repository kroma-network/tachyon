#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key_type_traits.h"

using namespace tachyon;

namespace {

using PKeyImpl = c::zk::plonk::bn254::ProvingKeyImpl;

}  // namespace

tachyon_bn254_plonk_proving_key*
tachyon_bn254_plonk_proving_key_create_from_state(const uint8_t* state,
                                                  size_t state_len) {
  PKeyImpl* pkey = new PKeyImpl(absl::Span<const uint8_t>(state, state_len),
                                /*read_only_vk=*/false);
  return c::base::c_cast(pkey);
}

void tachyon_bn254_plonk_proving_key_destroy(
    tachyon_bn254_plonk_proving_key* pk) {
  delete c::base::native_cast(pk);
}

const tachyon_bn254_plonk_verifying_key*
tachyon_bn254_plonk_proving_key_get_verifying_key(
    const tachyon_bn254_plonk_proving_key* pk) {
  const PKeyImpl* pkey = c::base::native_cast(pk);
  return reinterpret_cast<const tachyon_bn254_plonk_verifying_key*>(
      &pkey->verifying_key());
}
