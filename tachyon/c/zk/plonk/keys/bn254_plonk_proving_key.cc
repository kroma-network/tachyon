#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

#include "tachyon/c/zk/plonk/halo2/bn254_halo2_ls.h"
#include "tachyon/c/zk/plonk/halo2/bn254_log_derivative_halo2_ls.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key_type_traits.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl.h"
#include "tachyon/zk/plonk/halo2/ls_type.h"

using namespace tachyon;

namespace {

using Halo2LS = c::zk::plonk::halo2::bn254::Halo2LS;
using LogDerivativeHalo2LS = c::zk::plonk::halo2::bn254::LogDerivativeHalo2LS;

template <typename LS>
using ScrollPKeyImpl =
    c::zk::plonk::ProvingKeyImpl<zk::plonk::halo2::Vendor::kScroll, LS>;

template <typename NativePKey>
void Destroy(const NativePKey* pk) {
  delete pk;
}

template <typename NativePKey>
const tachyon_bn254_plonk_verifying_key* GetVerifyingKey(const NativePKey* pk) {
  return c::base::c_cast(&pk->verifying_key());
}

}  // namespace

#define INVOKE(Method, ...)                                                   \
  switch (static_cast<zk::plonk::halo2::LSType>(pk->ls_type)) {               \
    case zk::plonk::halo2::LSType::kHalo2: {                                  \
      return Method(reinterpret_cast<ScrollPKeyImpl<Halo2LS>*>(pk->extra),    \
                    ##__VA_ARGS__);                                           \
    }                                                                         \
    case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {                     \
      return Method(                                                          \
          reinterpret_cast<ScrollPKeyImpl<LogDerivativeHalo2LS>*>(pk->extra), \
          ##__VA_ARGS__);                                                     \
    }                                                                         \
  }                                                                           \
  NOTREACHED()

tachyon_bn254_plonk_proving_key*
tachyon_bn254_plonk_scroll_proving_key_create_from_state(uint8_t ls_type,
                                                         const uint8_t* state,
                                                         size_t state_len) {
  tachyon_bn254_plonk_proving_key* pkey = new tachyon_bn254_plonk_proving_key;
  pkey->ls_type = ls_type;
  switch (static_cast<zk::plonk::halo2::LSType>(ls_type)) {
    case zk::plonk::halo2::LSType::kHalo2: {
      pkey->extra = new ScrollPKeyImpl<Halo2LS>(
          absl::Span<const uint8_t>(state, state_len),
          /*read_only_vk=*/false);
      return pkey;
    }
    case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
      pkey->extra = new ScrollPKeyImpl<LogDerivativeHalo2LS>(
          absl::Span<const uint8_t>(state, state_len),
          /*read_only_vk=*/false);
      return pkey;
    }
  }
  NOTREACHED();
  return nullptr;
}

void tachyon_bn254_plonk_scroll_proving_key_destroy(
    tachyon_bn254_plonk_proving_key* pk) {
  INVOKE(Destroy);
}

const tachyon_bn254_plonk_verifying_key*
tachyon_bn254_plonk_scroll_proving_key_get_verifying_key(
    const tachyon_bn254_plonk_proving_key* pk) {
  INVOKE(GetVerifyingKey);
}
