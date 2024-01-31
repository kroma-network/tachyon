#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_prover.h"

#include <string.h>

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"
#include "tachyon/c/zk/plonk/halo2/shplonk_prover_impl_base.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl_base.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/prover.h"

using namespace tachyon;

namespace {

class Bn254SHPlonkProverImpl
    : public c::zk::plonk::halo2::SHPlonkProverImplBase {
 public:
  Bn254SHPlonkProverImpl(
      base::OnceCallback<zk::plonk::halo2::Prover<PCS>()> callback,
      uint8_t transcript_type)
      : c::zk::plonk::halo2::SHPlonkProverImplBase(std::move(callback)),
        transcript_type_(transcript_type) {}

  uint8_t transcript_type() const { return transcript_type_; }

 private:
  uint8_t transcript_type_;
};

using PCS = c::zk::plonk::halo2::PCS;
using CS = zk::plonk::ConstraintSystem<PCS::Field>;
using ProverImpl = Bn254SHPlonkProverImpl;
using ProvingKey =
    c::zk::ProvingKeyImplBase<PCS::Poly, PCS::Evals, PCS::Commitment>;
using Data = zk::plonk::halo2::ArgumentData<PCS::Poly, PCS::Evals>;

}  // namespace

tachyon_halo2_bn254_shplonk_prover*
tachyon_halo2_bn254_shplonk_prover_create_from_unsafe_setup(
    uint8_t transcript_type, uint32_t k, const tachyon_bn254_fr* s) {
  math::bn254::BN254Curve::Init();

  ProverImpl* prover = new ProverImpl(
      [transcript_type, k, s]() {
        PCS pcs;
        size_t n = size_t{1} << k;
        math::bn254::Fr::BigIntTy bigint;
        memcpy(bigint.limbs, reinterpret_cast<const uint8_t*>(s->limbs),
               sizeof(uint64_t) * math::bn254::Fr::kLimbNums);
        CHECK(pcs.UnsafeSetup(n, math::bn254::Fr::FromMontgomery(bigint)));
        base::Uint8VectorBuffer write_buf;
        std::unique_ptr<crypto::TranscriptWriter<math::bn254::G1AffinePoint>>
            writer;
        if (transcript_type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
          writer = std::make_unique<
              zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>(
              std::move(write_buf));
        } else {
          NOTREACHED();
        }
        zk::plonk::halo2::Prover<PCS> prover =
            zk::plonk::halo2::Prover<PCS>::CreateFromRNG(
                std::move(pcs), std::move(writer),
                /*rng=*/nullptr,
                /*blinding_factors=*/0);
        prover.set_domain(PCS::Domain::Create(n));
        return prover;
      },
      transcript_type);
  return reinterpret_cast<tachyon_halo2_bn254_shplonk_prover*>(prover);
}

void tachyon_halo2_bn254_shplonk_prover_destroy(
    tachyon_halo2_bn254_shplonk_prover* prover) {
  delete reinterpret_cast<ProverImpl*>(prover);
}

uint32_t tachyon_halo2_bn254_shplonk_prover_get_k(
    const tachyon_halo2_bn254_shplonk_prover* prover) {
  return reinterpret_cast<const ProverImpl*>(prover)->pcs().K();
}

size_t tachyon_halo2_bn254_shplonk_prover_get_n(
    const tachyon_halo2_bn254_shplonk_prover* prover) {
  return reinterpret_cast<const ProverImpl*>(prover)->pcs().N();
}

tachyon_bn254_blinder* tachyon_halo2_bn254_shplonk_prover_get_blinder(
    tachyon_halo2_bn254_shplonk_prover* prover) {
  return reinterpret_cast<tachyon_bn254_blinder*>(
      &(reinterpret_cast<ProverImpl*>(prover)->blinder()));
}

const tachyon_bn254_univariate_evaluation_domain*
tachyon_halo2_bn254_shplonk_prover_get_domain(
    const tachyon_halo2_bn254_shplonk_prover* prover) {
  return reinterpret_cast<const tachyon_bn254_univariate_evaluation_domain*>(
      reinterpret_cast<const ProverImpl*>(prover)->domain());
}

tachyon_bn254_g1_jacobian* tachyon_halo2_bn254_shplonk_prover_commit(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly) {
  return reinterpret_cast<const ProverImpl*>(prover)->Commit(
      reinterpret_cast<const PCS::Domain::DensePoly&>(*poly)
          .coefficients()
          .coefficients());
}

tachyon_bn254_g1_jacobian* tachyon_halo2_bn254_shplonk_prover_commit_lagrange(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_univariate_evaluations* evals) {
  return reinterpret_cast<const ProverImpl*>(prover)->CommitLagrange(
      reinterpret_cast<const PCS::Domain::Evals&>(*evals).evaluations());
}

void tachyon_halo2_bn254_shplonk_prover_set_rng_state(
    tachyon_halo2_bn254_shplonk_prover* prover, const uint8_t* state,
    size_t state_len) {
  base::Buffer buffer(const_cast<uint8_t*>(state), state_len);
  uint32_t x, y, z, w;
  CHECK(buffer.Read32LE(&x));
  CHECK(buffer.Read32LE(&y));
  CHECK(buffer.Read32LE(&z));
  CHECK(buffer.Read32LE(&w));
  CHECK(buffer.Done());
  reinterpret_cast<ProverImpl*>(prover)->SetRng(
      std::make_unique<crypto::XORShiftRNG>(
          crypto::XORShiftRNG::FromState(x, y, z, w)));
}

void tachyon_halo2_bn254_shplonk_prover_set_transcript_state(
    tachyon_halo2_bn254_shplonk_prover* prover, const uint8_t* state,
    size_t state_len) {
  ProverImpl* prover_impl = reinterpret_cast<ProverImpl*>(prover);
  uint8_t transcript_type = prover_impl->transcript_type();
  base::Uint8VectorBuffer write_buf;
  if (transcript_type == TACHYON_HALO2_BLAKE_TRANSCRIPT) {
    std::unique_ptr<zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>
        writer = std::make_unique<
            zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>(
            std::move(write_buf));
    writer->SetState(absl::Span<const uint8_t>(state, state_len));
    prover_impl->SetTranscript(std::move(writer));
  } else {
    NOTREACHED();
  }
}

void tachyon_halo2_bn254_shplonk_prover_set_extended_domain(
    tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_plonk_proving_key* pk) {
  const tachyon_bn254_plonk_verifying_key* vk =
      tachyon_bn254_plonk_proving_key_get_verifying_key(pk);
  const tachyon_bn254_plonk_constraint_system* cs =
      tachyon_bn254_plonk_verifying_key_get_constraint_system(vk);
  uint32_t extended_k = reinterpret_cast<const CS*>(cs)->ComputeExtendedK(
      reinterpret_cast<const ProverImpl*>(prover)->pcs().K());
  reinterpret_cast<ProverImpl*>(prover)->set_extended_domain(
      PCS::ExtendedDomain::Create(size_t{1} << extended_k));
}

void tachyon_halo2_bn254_shplonk_prover_create_proof(
    tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_plonk_proving_key* pk,
    tachyon_halo2_bn254_argument_data* data) {
  reinterpret_cast<ProverImpl*>(prover)->CreateProof(
      reinterpret_cast<const ProvingKey&>(*pk), reinterpret_cast<Data*>(data));
}

void tachyon_halo2_bn254_shplonk_prover_get_proof(
    const tachyon_halo2_bn254_shplonk_prover* prover, uint8_t* proof,
    size_t* proof_len) {
  const crypto::TranscriptWriter<PCS::Commitment>* transcript =
      reinterpret_cast<const ProverImpl*>(prover)->GetWriter();
  const std::vector<uint8_t>& buffer = transcript->buffer().owned_buffer();
  *proof_len = buffer.size();
  if (proof == nullptr) return;
  memcpy(proof, buffer.data(), buffer.size());
}

void tachyon_halo2_bn254_shplonk_prover_set_transcript_repr(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    tachyon_bn254_plonk_proving_key* pk) {
  reinterpret_cast<ProvingKey*>(pk)->SetTranscriptRepr(
      reinterpret_cast<const ProverImpl&>(*prover));
}
