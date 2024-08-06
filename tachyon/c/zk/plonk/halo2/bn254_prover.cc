#include "tachyon/c/zk/plonk/halo2/bn254_prover.h"

#include <string.h>

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"
#include "tachyon/c/zk/base/bn254_blinder_type_traits.h"
#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_argument_data_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_gwc_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_halo2_ls.h"
#include "tachyon/c/zk/plonk/halo2/bn254_log_derivative_halo2_ls.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"
#include "tachyon/c/zk/plonk/halo2/kzg_family_prover_impl.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl.h"
#include "tachyon/crypto/random/cha_cha20/cha_cha20_rng.h"
#include "tachyon/crypto/random/rng_type.h"
#include "tachyon/crypto/random/xor_shift/xor_shift_rng.h"
#include "tachyon/math/elliptic_curves/bn/bn254/halo2/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/ls_type.h"
#include "tachyon/zk/plonk/halo2/pcs_type.h"
#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/prover.h"
#include "tachyon/zk/plonk/halo2/sha256_transcript.h"
#include "tachyon/zk/plonk/halo2/snark_verifier_poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/transcript_type.h"

using namespace tachyon;

using GWCPCS = c::zk::plonk::halo2::bn254::GWCPCS;
using SHPlonkPCS = c::zk::plonk::halo2::bn254::SHPlonkPCS;
using Halo2LS = c::zk::plonk::halo2::bn254::Halo2LS;
using LogDerivativeHalo2LS = c::zk::plonk::halo2::bn254::LogDerivativeHalo2LS;
using XORShiftRNG = crypto::XORShiftRNG;
using ChaCha20RNG = crypto::ChaCha20RNG;

template <typename PCS, typename LS>
using ProverImpl = c::zk::plonk::halo2::KZGFamilyProverImpl<PCS, LS>;
template <typename LS>
using ScrollProvingKey =
    c::zk::plonk::ProvingKeyImpl<zk::plonk::halo2::Vendor::kScroll, LS>;

namespace {

template <typename PCS, typename LS>
zk::plonk::halo2::Prover<PCS, LS> CreateProver(uint8_t transcript_type,
                                               uint32_t k,
                                               const tachyon_bn254_fr* s) {
  PCS pcs;
  size_t n = size_t{1} << k;
  math::bn254::Fr::BigIntTy bigint;
  memcpy(bigint.limbs, reinterpret_cast<const uint8_t*>(s->limbs),
         sizeof(uint64_t) * math::bn254::Fr::kLimbNums);
  CHECK(pcs.UnsafeSetup(n, math::bn254::Fr::FromMontgomery(bigint)));
  base::Uint8VectorBuffer write_buf;
  std::unique_ptr<crypto::TranscriptWriter<math::bn254::G1AffinePoint>> writer;
  switch (static_cast<zk::plonk::halo2::TranscriptType>(transcript_type)) {
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
    case zk::plonk::halo2::TranscriptType::kSnarkVerifierPoseidon: {
      writer = std::make_unique<zk::plonk::halo2::SnarkVerifierPoseidonWriter<
          math::bn254::G1AffinePoint>>(std::move(write_buf));
      break;
    }
  }
  CHECK(writer);
  zk::plonk::halo2::Prover<PCS, LS> prover =
      zk::plonk::halo2::Prover<PCS, LS>::Create(std::move(pcs),
                                                std::move(writer),
                                                /*rng=*/nullptr,
                                                /*blinding_factors=*/0);
  prover.set_domain(PCS::Domain::Create(n));
  return prover;
}

template <typename PCS, typename LS>
zk::plonk::halo2::Prover<PCS, LS> CreateProverFromParams(
    uint8_t transcript_type, uint32_t k, const uint8_t* params,
    size_t params_len) {
  PCS pcs;
  size_t n = size_t{1} << k;
  base::ReadOnlyBuffer read_buf(params, params_len);
  c::zk::plonk::ReadBuffer(read_buf, pcs);

  base::Uint8VectorBuffer write_buf;
  std::unique_ptr<crypto::TranscriptWriter<math::bn254::G1AffinePoint>> writer;
  switch (static_cast<zk::plonk::halo2::TranscriptType>(transcript_type)) {
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
    case zk::plonk::halo2::TranscriptType::kSnarkVerifierPoseidon: {
      writer = std::make_unique<zk::plonk::halo2::SnarkVerifierPoseidonWriter<
          math::bn254::G1AffinePoint>>(std::move(write_buf));
      break;
    }
  }
  CHECK(writer);
  zk::plonk::halo2::Prover<PCS, LS> prover =
      zk::plonk::halo2::Prover<PCS, LS>::Create(std::move(pcs),
                                                std::move(writer),
                                                /*rng=*/nullptr,
                                                /*blinding_factors=*/0);
  prover.set_domain(PCS::Domain::Create(n));
  return prover;
}

template <typename NativeProver>
void Destroy(NativeProver* prover) {
  delete prover;
}

template <typename NativeEntity>
uint32_t GetK(NativeEntity* entity) {
  return entity->pcs().K();
}

template <typename NativeEntity>
size_t GetN(NativeEntity* entity) {
  return entity->pcs().N();
}

template <typename NativeEntity>
const tachyon_bn254_g2_affine* GetSG2(NativeEntity* entity) {
  return &c::base::c_cast(entity->pcs().SG2());
}

template <typename NativeProver>
tachyon_bn254_blinder* GetBlinder(NativeProver* prover) {
  return &c::base::c_cast(prover->blinder());
}

template <typename NativeProver>
const tachyon_bn254_univariate_evaluation_domain* GetDomain(
    NativeProver* prover) {
  return c::base::c_cast(prover->domain());
}

template <typename NativeProver>
tachyon_bn254_g1_projective* Commit(
    NativeProver* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly) {
  const std::vector<math::bn254::Fr>& scalars =
      c::base::native_cast(*poly).coefficients().coefficients();
  return prover->CommitRaw(scalars);
}

template <typename NativeProver>
tachyon_bn254_g1_projective* CommitLagrange(
    NativeProver* prover, const tachyon_bn254_univariate_evaluations* evals) {
  const std::vector<math::bn254::Fr>& scalars =
      c::base::native_cast(*evals).evaluations();
  return prover->CommitLagrangeRaw(scalars);
}

template <typename NativeProver>
void BatchStart(NativeProver* prover, size_t len) {
  using PCS = typename NativeProver::PCS;

  if constexpr (PCS::kSupportsBatchMode) {
    prover->pcs().SetBatchMode(len);
  } else {
    NOTREACHED() << "PCS doesn't support batch commitment";
  }
}

template <typename NativeProver>
void BatchCommit(NativeProver* prover,
                 const tachyon_bn254_univariate_dense_polynomial* poly,
                 size_t idx) {
  using PCS = typename NativeProver::PCS;

  if constexpr (PCS::kSupportsBatchMode) {
    prover->BatchCommitAt(c::base::native_cast(*poly), idx);
  } else {
    NOTREACHED() << "PCS doesn't support batch commitment";
  }
}

template <typename NativeProver>
void BatchCommitLagrange(NativeProver* prover,
                         const tachyon_bn254_univariate_evaluations* evals,
                         size_t idx) {
  using PCS = typename NativeProver::PCS;

  if constexpr (PCS::kSupportsBatchMode) {
    prover->BatchCommitAt(c::base::native_cast(*evals), idx);
  } else {
    NOTREACHED() << "PCS doesn't support batch commitment";
  }
}

template <typename NativeProver>
void BatchEnd(NativeProver* prover, tachyon_bn254_g1_affine* points,
              size_t len) {
  using PCS = typename NativeProver::PCS;
  using Commitment = typename PCS::Commitment;

  if constexpr (PCS::kSupportsBatchMode) {
    std::vector<Commitment> commitments = prover->pcs().GetBatchCommitments();
    CHECK_EQ(commitments.size(), len);
    // TODO(chokobole): Remove this |memcpy()| by modifying
    // |GetBatchCommitments()| to take the out parameters |points|.
    memcpy(points, commitments.data(), len * sizeof(Commitment));
  } else {
    NOTREACHED() << "PCS doesn't support batch commitment";
  }
}

template <typename NativeProver>
void SetRngState(NativeProver* prover, uint8_t rng_type, const uint8_t* state,
                 size_t state_len) {
  std::unique_ptr<crypto::RNG> rng;
  switch (static_cast<crypto::RNGType>(rng_type)) {
    case crypto::RNGType::kXORShift:
      rng = std::make_unique<crypto::XORShiftRNG>();
      break;
    case crypto::RNGType::kChaCha20:
      rng = std::make_unique<crypto::ChaCha20RNG>();
      break;
  }
  CHECK(rng);
  base::ReadOnlyBuffer buffer(state, state_len);
  CHECK(rng->ReadFromBuffer(buffer));
  prover->SetRng(std::move(rng));
}

template <typename NativeProver>
void SetTranscriptState(NativeProver* prover, const uint8_t* state,
                        size_t state_len) {
  uint8_t transcript_type = prover->transcript_type();
  base::Uint8VectorBuffer write_buf;
  switch (static_cast<zk::plonk::halo2::TranscriptType>(transcript_type)) {
    case zk::plonk::halo2::TranscriptType::kBlake2b: {
      auto writer = std::make_unique<
          zk::plonk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>(
          std::move(write_buf));
      absl::Span<const uint8_t> state_span(state, state_len);
      writer->SetState(state_span);
      prover->SetTranscript(state_span, std::move(writer));
      return;
    }
    case zk::plonk::halo2::TranscriptType::kPoseidon: {
      auto writer = std::make_unique<
          zk::plonk::halo2::PoseidonWriter<math::bn254::G1AffinePoint>>(
          std::move(write_buf));
      absl::Span<const uint8_t> state_span(state, state_len);
      writer->SetState(state_span);
      prover->SetTranscript(state_span, std::move(writer));
      return;
    }
    case zk::plonk::halo2::TranscriptType::kSha256: {
      auto writer = std::make_unique<
          zk::plonk::halo2::Sha256Writer<math::bn254::G1AffinePoint>>(
          std::move(write_buf));
      absl::Span<const uint8_t> state_span(state, state_len);
      writer->SetState(state_span);
      prover->SetTranscript(state_span, std::move(writer));
      return;
    }
    case zk::plonk::halo2::TranscriptType::kSnarkVerifierPoseidon: {
      auto writer =
          std::make_unique<zk::plonk::halo2::SnarkVerifierPoseidonWriter<
              math::bn254::G1AffinePoint>>(std::move(write_buf));
      absl::Span<const uint8_t> state_span(state, state_len);
      writer->SetState(state_span);
      prover->SetTranscript(state_span, std::move(writer));
      return;
    }
  }
  NOTREACHED();
}

template <typename NativeProver>
void SetExtendedDomain(NativeProver* prover,
                       const tachyon_bn254_plonk_proving_key* pk) {
  using PCS = typename NativeProver::PCS;

  const tachyon_bn254_plonk_verifying_key* vk =
      tachyon_bn254_plonk_scroll_proving_key_get_verifying_key(pk);
  const tachyon_bn254_plonk_constraint_system* cs =
      tachyon_bn254_plonk_verifying_key_get_constraint_system(vk);

  uint32_t extended_k =
      c::base::native_cast(cs)->ComputeExtendedK(prover->pcs().K());
  prover->set_extended_domain(
      PCS::ExtendedDomain::Create(size_t{1} << extended_k));
#if TACHYON_CUDA
  prover->EnableIcicleNTT();
#endif
}

template <typename NativeProver, typename NativeProvingKey>
void CreateProof(NativeProver* prover, NativeProvingKey* pk,
                 tachyon_halo2_bn254_argument_data* data) {
  prover->CreateProof(*pk, c::base::native_cast(data));
}

template <typename NativeProver>
void GetProof(NativeProver* prover, uint8_t* proof, size_t* proof_len) {
  const crypto::TranscriptWriter<math::bn254::G1AffinePoint>* transcript =
      prover->GetWriter();
  const std::vector<uint8_t>& buffer = transcript->buffer().owned_buffer();
  *proof_len = buffer.size();
  if (proof == nullptr) return;
  memcpy(proof, buffer.data(), buffer.size());
}

}  // namespace

#define INVOKE_PROVER(Method, ...)                                             \
  switch (static_cast<zk::plonk::halo2::PCSType>(prover->pcs_type)) {          \
    case zk::plonk::halo2::PCSType::kGWC: {                                    \
      switch (static_cast<zk::plonk::halo2::LSType>(prover->ls_type)) {        \
        case zk::plonk::halo2::LSType::kHalo2: {                               \
          return Method(                                                       \
              reinterpret_cast<ProverImpl<GWCPCS, Halo2LS>*>(prover->extra),   \
              ##__VA_ARGS__);                                                  \
        }                                                                      \
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {                  \
          return Method(                                                       \
              reinterpret_cast<ProverImpl<GWCPCS, LogDerivativeHalo2LS>*>(     \
                  prover->extra),                                              \
              ##__VA_ARGS__);                                                  \
        }                                                                      \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case zk::plonk::halo2::PCSType::kSHPlonk: {                                \
      switch (static_cast<zk::plonk::halo2::LSType>(prover->ls_type)) {        \
        case zk::plonk::halo2::LSType::kHalo2: {                               \
          return Method(reinterpret_cast<ProverImpl<SHPlonkPCS, Halo2LS>*>(    \
                            prover->extra),                                    \
                        ##__VA_ARGS__);                                        \
        }                                                                      \
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {                  \
          return Method(                                                       \
              reinterpret_cast<ProverImpl<SHPlonkPCS, LogDerivativeHalo2LS>*>( \
                  prover->extra),                                              \
              ##__VA_ARGS__);                                                  \
        }                                                                      \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
  }                                                                            \
  NOTREACHED()

#define INVOKE_ENTITY(Method, ...)                                            \
  switch (static_cast<zk::plonk::halo2::PCSType>(prover->pcs_type)) {         \
    case zk::plonk::halo2::PCSType::kGWC: {                                   \
      return Method(reinterpret_cast<zk::Entity<GWCPCS>*>(prover->extra),     \
                    ##__VA_ARGS__);                                           \
    }                                                                         \
    case zk::plonk::halo2::PCSType::kSHPlonk: {                               \
      return Method(reinterpret_cast<zk::Entity<SHPlonkPCS>*>(prover->extra), \
                    ##__VA_ARGS__);                                           \
    }                                                                         \
  }                                                                           \
  NOTREACHED()

#define INVOKE_PROVER_BASE(Method, ...)                                       \
  switch (static_cast<zk::plonk::halo2::PCSType>(prover->pcs_type)) {         \
    case zk::plonk::halo2::PCSType::kGWC: {                                   \
      return Method(reinterpret_cast<zk::ProverBase<GWCPCS>*>(prover->extra), \
                    ##__VA_ARGS__);                                           \
    }                                                                         \
    case zk::plonk::halo2::PCSType::kSHPlonk: {                               \
      return Method(                                                          \
          reinterpret_cast<zk::ProverBase<SHPlonkPCS>*>(prover->extra),       \
          ##__VA_ARGS__);                                                     \
    }                                                                         \
  }                                                                           \
  NOTREACHED()

tachyon_halo2_bn254_prover* tachyon_halo2_bn254_prover_create_from_unsafe_setup(
    uint8_t pcs_type, uint8_t ls_type, uint8_t transcript_type, uint32_t k,
    const tachyon_bn254_fr* s) {
  tachyon_halo2_bn254_prover* prover = new tachyon_halo2_bn254_prover;
  prover->pcs_type = pcs_type;
  prover->ls_type = ls_type;
  math::bn254::BN254Curve::Init();
  math::halo2::OverrideSubgroupGenerator();

  switch (static_cast<zk::plonk::halo2::PCSType>(pcs_type)) {
    case zk::plonk::halo2::PCSType::kGWC: {
      switch (static_cast<zk::plonk::halo2::LSType>(ls_type)) {
        case zk::plonk::halo2::LSType::kHalo2: {
          prover->extra = new ProverImpl<GWCPCS, Halo2LS>(
              CreateProver<GWCPCS, Halo2LS>(transcript_type, k, s),
              transcript_type);
          return prover;
        }
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
          prover->extra = new ProverImpl<GWCPCS, LogDerivativeHalo2LS>(
              CreateProver<GWCPCS, LogDerivativeHalo2LS>(transcript_type, k, s),
              transcript_type);
          return prover;
        }
      }
      break;
    }
    case zk::plonk::halo2::PCSType::kSHPlonk: {
      switch (static_cast<zk::plonk::halo2::LSType>(ls_type)) {
        case zk::plonk::halo2::LSType::kHalo2: {
          prover->extra = new ProverImpl<SHPlonkPCS, Halo2LS>(
              CreateProver<SHPlonkPCS, Halo2LS>(transcript_type, k, s),
              transcript_type);
          return prover;
        }
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
          prover->extra = new ProverImpl<SHPlonkPCS, LogDerivativeHalo2LS>(
              CreateProver<SHPlonkPCS, LogDerivativeHalo2LS>(transcript_type, k,
                                                             s),
              transcript_type);
          return prover;
        }
      }
      break;
    }
  }
  NOTREACHED();
  return nullptr;
}

tachyon_halo2_bn254_prover* tachyon_halo2_bn254_prover_create_from_params(
    uint8_t pcs_type, uint8_t ls_type, uint8_t transcript_type, uint32_t k,
    const uint8_t* params, size_t params_len) {
  tachyon_halo2_bn254_prover* prover = new tachyon_halo2_bn254_prover;
  prover->pcs_type = pcs_type;
  prover->ls_type = ls_type;
  math::bn254::BN254Curve::Init();
  math::halo2::OverrideSubgroupGenerator();

  switch (static_cast<zk::plonk::halo2::PCSType>(pcs_type)) {
    case zk::plonk::halo2::PCSType::kGWC: {
      switch (static_cast<zk::plonk::halo2::LSType>(ls_type)) {
        case zk::plonk::halo2::LSType::kHalo2: {
          prover->extra = new ProverImpl<GWCPCS, Halo2LS>(
              CreateProverFromParams<GWCPCS, Halo2LS>(transcript_type, k,
                                                      params, params_len),
              transcript_type);
          return prover;
        }
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
          prover->extra = new ProverImpl<GWCPCS, LogDerivativeHalo2LS>(
              CreateProverFromParams<GWCPCS, LogDerivativeHalo2LS>(
                  transcript_type, k, params, params_len),
              transcript_type);
          return prover;
        }
      }
      break;
    }
    case zk::plonk::halo2::PCSType::kSHPlonk: {
      switch (static_cast<zk::plonk::halo2::LSType>(ls_type)) {
        case zk::plonk::halo2::LSType::kHalo2: {
          prover->extra = new ProverImpl<SHPlonkPCS, Halo2LS>(
              CreateProverFromParams<SHPlonkPCS, Halo2LS>(transcript_type, k,
                                                          params, params_len),
              transcript_type);
          return prover;
        }
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
          prover->extra = new ProverImpl<SHPlonkPCS, LogDerivativeHalo2LS>(
              CreateProverFromParams<SHPlonkPCS, LogDerivativeHalo2LS>(
                  transcript_type, k, params, params_len),
              transcript_type);
          return prover;
        }
      }
      break;
    }
  }
  NOTREACHED();
  return nullptr;
}

void tachyon_halo2_bn254_prover_destroy(tachyon_halo2_bn254_prover* prover) {
  INVOKE_PROVER(Destroy);
}

uint32_t tachyon_halo2_bn254_prover_get_k(
    const tachyon_halo2_bn254_prover* prover) {
  INVOKE_ENTITY(GetK);
  return 0;
}

size_t tachyon_halo2_bn254_prover_get_n(
    const tachyon_halo2_bn254_prover* prover) {
  INVOKE_ENTITY(GetN);
  return 0;
}

const tachyon_bn254_g2_affine* tachyon_halo2_bn254_prover_get_s_g2(
    const tachyon_halo2_bn254_prover* prover) {
  INVOKE_ENTITY(GetSG2);
  return nullptr;
}

tachyon_bn254_blinder* tachyon_halo2_bn254_prover_get_blinder(
    tachyon_halo2_bn254_prover* prover) {
  INVOKE_PROVER_BASE(GetBlinder);
  return nullptr;
}

const tachyon_bn254_univariate_evaluation_domain*
tachyon_halo2_bn254_prover_get_domain(
    const tachyon_halo2_bn254_prover* prover) {
  INVOKE_ENTITY(GetDomain);
  return nullptr;
}

tachyon_bn254_g1_projective* tachyon_halo2_bn254_prover_commit(
    const tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly) {
  INVOKE_PROVER(Commit, poly);
  return nullptr;
}

tachyon_bn254_g1_projective* tachyon_halo2_bn254_prover_commit_lagrange(
    const tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_univariate_evaluations* evals) {
  INVOKE_PROVER(CommitLagrange, evals);
  return nullptr;
}

void tachyon_halo2_bn254_prover_batch_start(
    const tachyon_halo2_bn254_prover* prover, size_t len) {
  INVOKE_PROVER(BatchStart, len);
}

void tachyon_halo2_bn254_prover_batch_commit(
    const tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly, size_t idx) {
  INVOKE_PROVER(BatchCommit, poly, idx);
}

void tachyon_halo2_bn254_prover_batch_commit_lagrange(
    const tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_univariate_evaluations* evals, size_t idx) {
  INVOKE_PROVER(BatchCommitLagrange, evals, idx);
}

void tachyon_halo2_bn254_prover_batch_end(
    const tachyon_halo2_bn254_prover* prover, tachyon_bn254_g1_affine* points,
    size_t len) {
  INVOKE_PROVER(BatchEnd, points, len);
}

void tachyon_halo2_bn254_prover_set_rng_state(
    tachyon_halo2_bn254_prover* prover, uint8_t rng_type, const uint8_t* state,
    size_t state_len) {
  INVOKE_PROVER(SetRngState, rng_type, state, state_len);
}

void tachyon_halo2_bn254_prover_set_transcript_state(
    tachyon_halo2_bn254_prover* prover, const uint8_t* state,
    size_t state_len) {
  INVOKE_PROVER(SetTranscriptState, state, state_len);
}

void tachyon_halo2_bn254_scroll_prover_set_extended_domain(
    tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_plonk_proving_key* pk) {
  INVOKE_PROVER(SetExtendedDomain, pk);
}

void tachyon_halo2_bn254_scroll_prover_create_proof(
    tachyon_halo2_bn254_prover* prover, tachyon_bn254_plonk_proving_key* pk,
    tachyon_halo2_bn254_argument_data* data) {
  switch (static_cast<zk::plonk::halo2::PCSType>(prover->pcs_type)) {
    case zk::plonk::halo2::PCSType::kGWC: {
      switch (static_cast<zk::plonk::halo2::LSType>(prover->ls_type)) {
        case zk::plonk::halo2::LSType::kHalo2: {
          CreateProof(
              reinterpret_cast<ProverImpl<GWCPCS, Halo2LS>*>(prover->extra),
              reinterpret_cast<ScrollProvingKey<Halo2LS>*>(pk->extra), data);
          return;
        }
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
          CreateProof(
              reinterpret_cast<ProverImpl<GWCPCS, LogDerivativeHalo2LS>*>(
                  prover->extra),
              reinterpret_cast<ScrollProvingKey<LogDerivativeHalo2LS>*>(
                  pk->extra),
              data);
          return;
        }
      }
      break;
    }
    case zk::plonk::halo2::PCSType::kSHPlonk: {
      switch (static_cast<zk::plonk::halo2::LSType>(prover->ls_type)) {
        case zk::plonk::halo2::LSType::kHalo2: {
          CreateProof(
              reinterpret_cast<ProverImpl<SHPlonkPCS, Halo2LS>*>(prover->extra),
              reinterpret_cast<ScrollProvingKey<Halo2LS>*>(pk->extra), data);
          return;
        }
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
          CreateProof(
              reinterpret_cast<ProverImpl<SHPlonkPCS, LogDerivativeHalo2LS>*>(
                  prover->extra),
              reinterpret_cast<ScrollProvingKey<LogDerivativeHalo2LS>*>(
                  pk->extra),
              data);
          return;
        }
      }
      break;
    }
  }
  NOTREACHED();
}

void tachyon_halo2_bn254_prover_get_proof(
    const tachyon_halo2_bn254_prover* prover, uint8_t* proof,
    size_t* proof_len) {
  INVOKE_PROVER_BASE(GetProof, proof, proof_len);
}

void tachyon_halo2_bn254_scroll_prover_set_transcript_repr(
    const tachyon_halo2_bn254_prover* prover,
    tachyon_bn254_plonk_proving_key* pk) {
  switch (static_cast<zk::plonk::halo2::PCSType>(prover->pcs_type)) {
    case zk::plonk::halo2::PCSType::kGWC: {
      switch (static_cast<zk::plonk::halo2::LSType>(prover->ls_type)) {
        case zk::plonk::halo2::LSType::kHalo2: {
          reinterpret_cast<ScrollProvingKey<Halo2LS>*>(pk->extra)
              ->SetTranscriptRepr(
                  *reinterpret_cast<zk::Entity<GWCPCS>*>(prover->extra));
          return;
        }
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
          reinterpret_cast<ScrollProvingKey<LogDerivativeHalo2LS>*>(pk->extra)
              ->SetTranscriptRepr(
                  *reinterpret_cast<zk::Entity<GWCPCS>*>(prover->extra));
          return;
        }
      }
      break;
    }
    case zk::plonk::halo2::PCSType::kSHPlonk: {
      switch (static_cast<zk::plonk::halo2::LSType>(prover->ls_type)) {
        case zk::plonk::halo2::LSType::kHalo2: {
          reinterpret_cast<ScrollProvingKey<Halo2LS>*>(pk->extra)
              ->SetTranscriptRepr(
                  *reinterpret_cast<zk::Entity<SHPlonkPCS>*>(prover->extra));
          return;
        }
        case zk::plonk::halo2::LSType::kLogDerivativeHalo2: {
          reinterpret_cast<ScrollProvingKey<LogDerivativeHalo2LS>*>(pk->extra)
              ->SetTranscriptRepr(
                  *reinterpret_cast<zk::Entity<SHPlonkPCS>*>(prover->extra));
          return;
        }
      }
      break;
    }
  }
  NOTREACHED();
}
