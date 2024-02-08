#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_verifier.h"

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_verifier_impl.h"
#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

using namespace tachyon;

using PCS = c::zk::plonk::halo2::bn254::PCS;
using Verifier = c::zk::plonk::halo2::bn254::SHPlonkVerifierImpl;
using VKey =
    zk::plonk::VerifyingKey<math::bn254::Fr, math::bn254::G1AffinePoint>;

tachyon_halo2_bn254_shplonk_verifier*
tachyon_halo2_bn254_shplonk_verifier_create_from_params(
    uint8_t transcript_type, uint32_t k, const uint8_t* params,
    size_t params_len, const uint8_t* proof, size_t proof_len) {
  math::bn254::BN254Curve::Init();

  return reinterpret_cast<tachyon_halo2_bn254_shplonk_verifier*>(new Verifier(
      [transcript_type, k, params, params_len, proof, proof_len]() {
        PCS pcs;
        base::ReadOnlyBuffer read_buf(params, params_len);
        CHECK(read_buf.Read(&pcs));

        read_buf = base::ReadOnlyBuffer(proof, proof_len);
        std::unique_ptr<crypto::TranscriptReader<math::bn254::G1AffinePoint>>
            reader;
        if (transcript_type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
          reader = std::make_unique<
              zk::plonk::halo2::Blake2bReader<math::bn254::G1AffinePoint>>(
              std::move(read_buf));
        } else {
          NOTREACHED();
        }
        zk::plonk::halo2::Verifier<PCS> verifier(std::move(pcs),
                                                 std::move(reader));
        verifier.set_domain(PCS::Domain::Create(size_t{1} << k));
        return verifier;
      },
      transcript_type));
}

void tachyon_halo2_bn254_shplonk_verifier_destroy(
    tachyon_halo2_bn254_shplonk_verifier* verifier) {
  delete reinterpret_cast<Verifier*>(verifier);
}

bool tachyon_halo2_bn254_shplonk_verifier_verify_proof(
    tachyon_halo2_bn254_shplonk_verifier* verifier,
    const tachyon_bn254_plonk_verifying_key* vkey,
    tachyon_halo2_bn254_instance_columns_vec* instance_columns_vec) {
  bool ret = reinterpret_cast<Verifier*>(verifier)->VerifyProof(
      reinterpret_cast<const VKey&>(*vkey),
      reinterpret_cast<std::vector<std::vector<std::vector<math::bn254::Fr>>>&>(
          *instance_columns_vec));
  tachyon_halo2_bn254_instance_columns_vec_destroy(instance_columns_vec);
  return ret;
}
