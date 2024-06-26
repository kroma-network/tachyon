#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_verifier.h"

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_verifier_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key_type_traits.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/sha256_transcript.h"
#include "tachyon/zk/plonk/halo2/transcript_type.h"

using namespace tachyon;

using PCS = c::zk::plonk::halo2::bn254::SHPlonkPCS;
using LS = c::zk::plonk::halo2::bn254::LS;
using Verifier = c::zk::plonk::halo2::VerifierImpl<PCS, LS>;

tachyon_halo2_bn254_shplonk_verifier*
tachyon_halo2_bn254_shplonk_verifier_create_from_params(
    uint8_t transcript_type, uint32_t k, const uint8_t* params,
    size_t params_len, const uint8_t* proof, size_t proof_len) {
  math::bn254::BN254Curve::Init();

  return c::base::c_cast(new Verifier(
      [transcript_type, k, params, params_len, proof, proof_len]() {
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
        }
        CHECK(reader);
        zk::plonk::halo2::Verifier<PCS, LS> verifier(std::move(pcs),
                                                     std::move(reader));
        verifier.set_domain(PCS::Domain::Create(size_t{1} << k));
        return verifier;
      },
      transcript_type));
}

void tachyon_halo2_bn254_shplonk_verifier_destroy(
    tachyon_halo2_bn254_shplonk_verifier* verifier) {
  delete c::base::native_cast(verifier);
}

bool tachyon_halo2_bn254_shplonk_verifier_verify_proof(
    tachyon_halo2_bn254_shplonk_verifier* verifier,
    const tachyon_bn254_plonk_verifying_key* vkey,
    tachyon_halo2_bn254_instance_columns_vec* instance_columns_vec) {
  bool ret = c::base::native_cast(verifier)->VerifyProof(
      c::base::native_cast(*vkey), c::base::native_cast(*instance_columns_vec));
  tachyon_halo2_bn254_instance_columns_vec_destroy(instance_columns_vec);
  return ret;
}
