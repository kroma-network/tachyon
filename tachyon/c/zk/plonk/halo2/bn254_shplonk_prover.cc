#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_prover.h"

#include <string.h>

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"
#include "tachyon/c/zk/base/bn254_blinder_type_traits.h"
#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_argument_data_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_ls.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"
#include "tachyon/c/zk/plonk/halo2/kzg_family_prover_impl.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl_base.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/prover.h"
#include "tachyon/zk/plonk/halo2/sha256_transcript.h"
#include "tachyon/zk/plonk/halo2/transcript_type.h"

using namespace tachyon;

using PCS = c::zk::plonk::halo2::bn254::SHPlonkPCS;
using LS = c::zk::plonk::halo2::bn254::LS;
using ProverImpl = c::zk::plonk::halo2::KZGFamilyProverImpl<PCS, LS>;
using ProvingKey = c::zk::plonk::ProvingKeyImplBase<LS>;

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
        switch (
            static_cast<zk::plonk::halo2::TranscriptType>(transcript_type)) {
          case zk::plonk::halo2::TranscriptType::kBlake2b: {
            writer = std::make_unique<
                zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>(
                std::move(write_buf));
            break;
          }
          case zk::plonk::halo2::TranscriptType::kPoseidon: {
            writer = std::make_unique<
                zk::plonk::halo2::PoseidonWriter<math::bn254::G1AffinePoint>>(
                std::move(write_buf));
            break;
          }
          case zk::plonk::halo2::TranscriptType::kSha256: {
            writer = std::make_unique<
                zk::plonk::halo2::Sha256Writer<math::bn254::G1AffinePoint>>(
                std::move(write_buf));
            break;
          }
        }
        CHECK(writer);
        zk::plonk::halo2::Prover<PCS, LS> prover =
            zk::plonk::halo2::Prover<PCS, LS>::CreateFromRNG(
                std::move(pcs), std::move(writer),
                /*rng=*/nullptr,
                /*blinding_factors=*/0);
        prover.set_domain(PCS::Domain::Create(n));
        return prover;
      },
      transcript_type);
  return reinterpret_cast<tachyon_halo2_bn254_shplonk_prover*>(prover);
}

tachyon_halo2_bn254_shplonk_prover*
tachyon_halo2_bn254_shplonk_prover_create_from_params(uint8_t transcript_type,
                                                      uint32_t k,
                                                      const uint8_t* params,
                                                      size_t params_len) {
  math::bn254::BN254Curve::Init();

  ProverImpl* prover = new ProverImpl(
      [transcript_type, k, params, params_len]() {
        PCS pcs;
        base::ReadOnlyBuffer read_buf(params, params_len);
        c::zk::plonk::ReadBuffer(read_buf, pcs);

        base::Uint8VectorBuffer write_buf;
        std::unique_ptr<crypto::TranscriptWriter<math::bn254::G1AffinePoint>>
            writer;
        switch (
            static_cast<zk::plonk::halo2::TranscriptType>(transcript_type)) {
          case zk::plonk::halo2::TranscriptType::kBlake2b: {
            writer = std::make_unique<
                zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>(
                std::move(write_buf));
            break;
          }
          case zk::plonk::halo2::TranscriptType::kPoseidon: {
            writer = std::make_unique<
                zk::plonk::halo2::PoseidonWriter<math::bn254::G1AffinePoint>>(
                std::move(write_buf));
            break;
          }
          case zk::plonk::halo2::TranscriptType::kSha256: {
            writer = std::make_unique<
                zk::plonk::halo2::Sha256Writer<math::bn254::G1AffinePoint>>(
                std::move(write_buf));
            break;
          }
        }
        CHECK(writer);
        zk::plonk::halo2::Prover<PCS, LS> prover =
            zk::plonk::halo2::Prover<PCS, LS>::CreateFromRNG(
                std::move(pcs), std::move(writer),
                /*rng=*/nullptr,
                /*blinding_factors=*/0);
        prover.set_domain(PCS::Domain::Create(size_t{1} << k));
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

const tachyon_bn254_g2_affine* tachyon_halo2_bn254_shplonk_prover_get_s_g2(
    const tachyon_halo2_bn254_shplonk_prover* prover) {
  return reinterpret_cast<const tachyon_bn254_g2_affine*>(
      &(reinterpret_cast<const ProverImpl*>(prover)->pcs().SG2()));
}

tachyon_bn254_blinder* tachyon_halo2_bn254_shplonk_prover_get_blinder(
    tachyon_halo2_bn254_shplonk_prover* prover) {
  return c::base::c_cast(&(reinterpret_cast<ProverImpl*>(prover)->blinder()));
}

const tachyon_bn254_univariate_evaluation_domain*
tachyon_halo2_bn254_shplonk_prover_get_domain(
    const tachyon_halo2_bn254_shplonk_prover* prover) {
  return c::base::c_cast(reinterpret_cast<const ProverImpl*>(prover)->domain());
}

tachyon_bn254_g1_jacobian* tachyon_halo2_bn254_shplonk_prover_commit(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly) {
  return reinterpret_cast<const ProverImpl*>(prover)->Commit(
      c::base::native_cast(*poly).coefficients().coefficients());
}

tachyon_bn254_g1_jacobian* tachyon_halo2_bn254_shplonk_prover_commit_lagrange(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_univariate_evaluations* evals) {
  return reinterpret_cast<const ProverImpl*>(prover)->CommitLagrange(
      c::base::native_cast(*evals).evaluations());
}

void tachyon_halo2_bn254_shplonk_prover_set_rng_state(
    tachyon_halo2_bn254_shplonk_prover* prover, const uint8_t* state,
    size_t state_len) {
  reinterpret_cast<ProverImpl*>(prover)->SetRngState(
      absl::Span<const uint8_t>(state, state_len));
}

void tachyon_halo2_bn254_shplonk_prover_set_transcript_state(
    tachyon_halo2_bn254_shplonk_prover* prover, const uint8_t* state,
    size_t state_len) {
  ProverImpl* prover_impl = reinterpret_cast<ProverImpl*>(prover);
  uint8_t transcript_type = prover_impl->transcript_type();
  base::Uint8VectorBuffer write_buf;
  switch (static_cast<zk::plonk::halo2::TranscriptType>(transcript_type)) {
    case zk::plonk::halo2::TranscriptType::kBlake2b: {
      std::unique_ptr<
          zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>
          writer = std::make_unique<
              zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>(
              std::move(write_buf));
      absl::Span<const uint8_t> state_span(state, state_len);
      writer->SetState(state_span);
      prover_impl->SetTranscript(state_span, std::move(writer));
      return;
    }
    case zk::plonk::halo2::TranscriptType::kPoseidon: {
      std::unique_ptr<
          zk::plonk::halo2::PoseidonWriter<math::bn254::G1AffinePoint>>
          writer = std::make_unique<
              zk::plonk::halo2::PoseidonWriter<math::bn254::G1AffinePoint>>(
              std::move(write_buf));
      absl::Span<const uint8_t> state_span(state, state_len);
      writer->SetState(state_span);
      prover_impl->SetTranscript(state_span, std::move(writer));
      return;
    }
    case zk::plonk::halo2::TranscriptType::kSha256: {
      std::unique_ptr<
          zk::plonk::halo2::Sha256Writer<math::bn254::G1AffinePoint>>
          writer = std::make_unique<
              zk::plonk::halo2::Sha256Writer<math::bn254::G1AffinePoint>>(
              std::move(write_buf));
      absl::Span<const uint8_t> state_span(state, state_len);
      writer->SetState(state_span);
      prover_impl->SetTranscript(state_span, std::move(writer));
      return;
    }
  }
  NOTREACHED();
}

void tachyon_halo2_bn254_shplonk_prover_set_extended_domain(
    tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_plonk_proving_key* pk) {
  const tachyon_bn254_plonk_verifying_key* vk =
      tachyon_bn254_plonk_proving_key_get_verifying_key(pk);
  const tachyon_bn254_plonk_constraint_system* cs =
      tachyon_bn254_plonk_verifying_key_get_constraint_system(vk);
  uint32_t extended_k = c::base::native_cast(cs)->ComputeExtendedK(
      reinterpret_cast<const ProverImpl*>(prover)->pcs().K());
  reinterpret_cast<ProverImpl*>(prover)->set_extended_domain(
      PCS::ExtendedDomain::Create(size_t{1} << extended_k));
}

void tachyon_halo2_bn254_shplonk_prover_create_proof(
    tachyon_halo2_bn254_shplonk_prover* prover,
    tachyon_bn254_plonk_proving_key* pk,
    tachyon_halo2_bn254_argument_data* data) {
  reinterpret_cast<ProverImpl*>(prover)->CreateProof(
      reinterpret_cast<ProvingKey&>(*pk), c::base::native_cast(data));
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
