#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

#include "tachyon/c/zk/plonk/halo2/bn254_ps.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key_type_traits.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl.h"
#include "tachyon/zk/plonk/halo2/pcs_type.h"

using namespace tachyon;

namespace {

using PSEGWC = c::zk::plonk::halo2::bn254::PSEGWC;
using PSESHPlonk = c::zk::plonk::halo2::bn254::PSESHPlonk;
using ScrollGWC = c::zk::plonk::halo2::bn254::ScrollGWC;
using ScrollSHPlonk = c::zk::plonk::halo2::bn254::ScrollSHPlonk;

template <typename PS>
using PKeyImpl = c::zk::plonk::ProvingKeyImpl<PS>;

template <typename NativePKey>
void Destroy(const NativePKey* pk) {
  delete pk;
}

template <typename NativePKey>
const tachyon_bn254_plonk_verifying_key* GetVerifyingKey(const NativePKey* pk) {
  return c::base::c_cast(&pk->verifying_key());
}

}  // namespace

#define INVOKE(Method, ...)                                                    \
  switch (static_cast<zk::plonk::halo2::Vendor>(pk->vendor)) {                 \
    case zk::plonk::halo2::Vendor::kPSE: {                                     \
      switch (static_cast<zk::plonk::halo2::PCSType>(pk->pcs_type)) {          \
        case zk::plonk::halo2::PCSType::kGWC: {                                \
          return Method(reinterpret_cast<PKeyImpl<PSEGWC>*>(pk->extra),        \
                        ##__VA_ARGS__);                                        \
        }                                                                      \
        case zk::plonk::halo2::PCSType::kSHPlonk: {                            \
          return Method(reinterpret_cast<PKeyImpl<PSESHPlonk>*>(pk->extra),    \
                        ##__VA_ARGS__);                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    case zk::plonk::halo2::Vendor::kScroll: {                                  \
      switch (static_cast<zk::plonk::halo2::PCSType>(pk->pcs_type)) {          \
        case zk::plonk::halo2::PCSType::kGWC: {                                \
          return Method(reinterpret_cast<PKeyImpl<ScrollGWC>*>(pk->extra),     \
                        ##__VA_ARGS__);                                        \
        }                                                                      \
        case zk::plonk::halo2::PCSType::kSHPlonk: {                            \
          return Method(reinterpret_cast<PKeyImpl<ScrollSHPlonk>*>(pk->extra), \
                        ##__VA_ARGS__);                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  NOTREACHED()

tachyon_bn254_plonk_proving_key*
tachyon_bn254_plonk_proving_key_create_from_state(uint8_t vendor,
                                                  uint8_t pcs_type,
                                                  const uint8_t* state,
                                                  size_t state_len) {
  tachyon_bn254_plonk_proving_key* pkey = new tachyon_bn254_plonk_proving_key;
  pkey->vendor = vendor;
  pkey->pcs_type = pcs_type;
  switch (static_cast<zk::plonk::halo2::Vendor>(vendor)) {
    case zk::plonk::halo2::Vendor::kPSE: {
      switch (static_cast<zk::plonk::halo2::PCSType>(pcs_type)) {
        case zk::plonk::halo2::PCSType::kGWC: {
          pkey->extra =
              new PKeyImpl<PSEGWC>(absl::Span<const uint8_t>(state, state_len),
                                   /*read_only_vk=*/false);
          return pkey;
        }
        case zk::plonk::halo2::PCSType::kSHPlonk: {
          pkey->extra = new PKeyImpl<PSESHPlonk>(
              absl::Span<const uint8_t>(state, state_len),
              /*read_only_vk=*/false);
          return pkey;
        }
      }
    }
    case zk::plonk::halo2::Vendor::kScroll: {
      switch (static_cast<zk::plonk::halo2::PCSType>(pcs_type)) {
        case zk::plonk::halo2::PCSType::kGWC: {
          pkey->extra = new PKeyImpl<ScrollGWC>(
              absl::Span<const uint8_t>(state, state_len),
              /*read_only_vk=*/false);
          return pkey;
        }
        case zk::plonk::halo2::PCSType::kSHPlonk: {
          pkey->extra = new PKeyImpl<ScrollSHPlonk>(
              absl::Span<const uint8_t>(state, state_len),
              /*read_only_vk=*/false);
          return pkey;
        }
      }
    }
  }
  NOTREACHED();
  return nullptr;
}

void tachyon_bn254_plonk_proving_key_destroy(
    tachyon_bn254_plonk_proving_key* pk) {
  INVOKE(Destroy);
}

const tachyon_bn254_plonk_verifying_key*
tachyon_bn254_plonk_proving_key_get_verifying_key(
    const tachyon_bn254_plonk_proving_key* pk) {
  INVOKE(GetVerifyingKey);
}
