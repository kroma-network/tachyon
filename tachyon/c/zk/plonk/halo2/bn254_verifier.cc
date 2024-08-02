#include "tachyon/c/zk/plonk/halo2/bn254_verifier.h"

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_ps.h"
#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"
#include "tachyon/c/zk/plonk/halo2/verifier_impl.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key_type_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/halo2/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/pcs_type.h"
#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/sha256_transcript.h"
#include "tachyon/zk/plonk/halo2/snark_verifier_poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/transcript_type.h"
#include "tachyon/zk/plonk/halo2/vendor.h"

using namespace tachyon;

using PSEGWC = c::zk::plonk::halo2::bn254::PSEGWC;
using PSESHPlonk = c::zk::plonk::halo2::bn254::PSESHPlonk;
using ScrollGWC = c::zk::plonk::halo2::bn254::ScrollGWC;
using ScrollSHPlonk = c::zk::plonk::halo2::bn254::ScrollSHPlonk;

template <typename PS>
using Verifier = c::zk::plonk::halo2::VerifierImpl<PS>;

namespace {

template <typename PS>
Verifier<PS>* CreateVerifierFromParams(uint8_t transcript_type, uint32_t k,
                                       const uint8_t* params, size_t params_len,
                                       const uint8_t* proof, size_t proof_len) {
  return new Verifier<PS>(
      [transcript_type, k, params, params_len, proof, proof_len]() {
        using PCS = typename PS::PCS;

        PCS pcs;
        base::ReadOnlyBuffer read_buf(params, params_len);
        CHECK(read_buf.Read(&pcs));

        read_buf = base::ReadOnlyBuffer(proof, proof_len);
        std::unique_ptr<crypto::TranscriptReader<math::bn254::G1AffinePoint>>
            reader;
        switch (
            static_cast<zk::plonk::halo2::TranscriptType>(transcript_type)) {
          case zk::plonk::halo2::TranscriptType::kBlake2b: {
            reader = std::make_unique<
                zk::plonk::halo2::Blake2bReader<math::bn254::G1AffinePoint>>(
                std::move(read_buf));
            break;
          }
          case zk::plonk::halo2::TranscriptType::kPoseidon: {
            reader = std::make_unique<
                zk::plonk::halo2::PoseidonReader<math::bn254::G1AffinePoint>>(
                std::move(read_buf));
            break;
          }
          case zk::plonk::halo2::TranscriptType::kSha256: {
            reader = std::make_unique<
                zk::plonk::halo2::Sha256Reader<math::bn254::G1AffinePoint>>(
                std::move(read_buf));
            break;
          }
          case zk::plonk::halo2::TranscriptType::kSnarkVerifierPoseidon: {
            reader =
                std::make_unique<zk::plonk::halo2::SnarkVerifierPoseidonReader<
                    math::bn254::G1AffinePoint>>(std::move(read_buf));
            break;
          }
        }
        CHECK(reader);
        zk::plonk::halo2::Verifier<PS> verifier(std::move(pcs),
                                                std::move(reader));
        verifier.set_domain(PCS::Domain::Create(size_t{1} << k));
        return verifier;
      },
      transcript_type);
}

template <typename NativeVerifier>
void Destroy(NativeVerifier* verifier) {
  delete verifier;
}

template <typename NativeVerifier>
bool VerifyProof(
    NativeVerifier* verifier, const tachyon_bn254_plonk_verifying_key* vkey,
    tachyon_halo2_bn254_instance_columns_vec* instance_columns_vec) {
  bool ret = verifier->VerifyProof(c::base::native_cast(*vkey),
                                   c::base::native_cast(*instance_columns_vec));
  tachyon_halo2_bn254_instance_columns_vec_destroy(instance_columns_vec);
  return ret;
}

}  // namespace

#define INVOKE_VERIFIER(Method, ...)                                          \
  switch (static_cast<zk::plonk::halo2::Vendor>(verifier->vendor)) {          \
    case zk::plonk::halo2::Vendor::kPSE: {                                    \
      switch (static_cast<zk::plonk::halo2::PCSType>(verifier->pcs_type)) {   \
        case zk::plonk::halo2::PCSType::kGWC: {                               \
          return Method(reinterpret_cast<Verifier<PSEGWC>*>(verifier->extra), \
                        ##__VA_ARGS__);                                       \
        }                                                                     \
        case zk::plonk::halo2::PCSType::kSHPlonk: {                           \
          return Method(                                                      \
              reinterpret_cast<Verifier<PSESHPlonk>*>(verifier->extra),       \
              ##__VA_ARGS__);                                                 \
        }                                                                     \
      }                                                                       \
      break;                                                                  \
    }                                                                         \
    case zk::plonk::halo2::Vendor::kScroll: {                                 \
      switch (static_cast<zk::plonk::halo2::PCSType>(verifier->pcs_type)) {   \
        case zk::plonk::halo2::PCSType::kGWC: {                               \
          return Method(                                                      \
              reinterpret_cast<Verifier<ScrollGWC>*>(verifier->extra),        \
              ##__VA_ARGS__);                                                 \
        }                                                                     \
        case zk::plonk::halo2::PCSType::kSHPlonk: {                           \
          return Method(                                                      \
              reinterpret_cast<Verifier<ScrollSHPlonk>*>(verifier->extra),    \
              ##__VA_ARGS__);                                                 \
        }                                                                     \
      }                                                                       \
      break;                                                                  \
    }                                                                         \
  }                                                                           \
  NOTREACHED()

tachyon_halo2_bn254_verifier* tachyon_halo2_bn254_verifier_create_from_params(
    uint8_t vendor, uint8_t pcs_type, uint8_t transcript_type, uint32_t k,
    const uint8_t* params, size_t params_len, const uint8_t* proof,
    size_t proof_len) {
  tachyon_halo2_bn254_verifier* verifier = new tachyon_halo2_bn254_verifier;
  verifier->vendor = vendor;
  verifier->pcs_type = pcs_type;
  math::bn254::BN254Curve::Init();
  math::halo2::OverrideSubgroupGenerator();

  switch (static_cast<zk::plonk::halo2::Vendor>(vendor)) {
    case zk::plonk::halo2::Vendor::kPSE: {
      switch (static_cast<zk::plonk::halo2::PCSType>(pcs_type)) {
        case zk::plonk::halo2::PCSType::kGWC: {
          verifier->extra = CreateVerifierFromParams<PSEGWC>(
              transcript_type, k, params, params_len, proof, proof_len);
          return verifier;
        }
        case zk::plonk::halo2::PCSType::kSHPlonk: {
          verifier->extra = CreateVerifierFromParams<PSESHPlonk>(
              transcript_type, k, params, params_len, proof, proof_len);
          return verifier;
        }
      }
      break;
    }
    case zk::plonk::halo2::Vendor::kScroll: {
      switch (static_cast<zk::plonk::halo2::PCSType>(pcs_type)) {
        case zk::plonk::halo2::PCSType::kGWC: {
          verifier->extra = CreateVerifierFromParams<ScrollGWC>(
              transcript_type, k, params, params_len, proof, proof_len);
          return verifier;
        }
        case zk::plonk::halo2::PCSType::kSHPlonk: {
          verifier->extra = CreateVerifierFromParams<ScrollSHPlonk>(
              transcript_type, k, params, params_len, proof, proof_len);
          return verifier;
        }
      }
      break;
    }
  }
  NOTREACHED();
  return nullptr;
}

void tachyon_halo2_bn254_verifier_destroy(
    tachyon_halo2_bn254_verifier* verifier) {
  INVOKE_VERIFIER(Destroy);
}

bool tachyon_halo2_bn254_verifier_verify_proof(
    tachyon_halo2_bn254_verifier* verifier,
    const tachyon_bn254_plonk_verifying_key* vkey,
    tachyon_halo2_bn254_instance_columns_vec* instance_columns_vec) {
  INVOKE_VERIFIER(VerifyProof, vkey, instance_columns_vec);
  return false;
}
