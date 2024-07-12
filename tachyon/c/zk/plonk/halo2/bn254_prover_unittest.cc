#include "tachyon/c/zk/plonk/halo2/bn254_prover.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/c/crypto/random/rng.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"
#include "tachyon/c/zk/base/bn254_blinder_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_gwc_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_halo2_ls.h"
#include "tachyon/c/zk/plonk/halo2/bn254_log_derivative_halo2_ls.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"
#include "tachyon/c/zk/plonk/halo2/kzg_family_prover_impl.h"
#include "tachyon/c/zk/plonk/halo2/test/bn254_halo2_params_data.h"
#include "tachyon/crypto/random/xor_shift/xor_shift_rng.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/ls_type.h"
#include "tachyon/zk/plonk/halo2/pcs_type.h"
#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"
#include "tachyon/zk/plonk/halo2/sha256_transcript.h"
#include "tachyon/zk/plonk/halo2/transcript_type.h"

namespace tachyon::zk::plonk::halo2::bn254 {

namespace {

using GWCPCS = c::zk::plonk::halo2::bn254::GWCPCS;
using SHPlonkPCS = c::zk::plonk::halo2::bn254::SHPlonkPCS;
using Halo2LS = c::zk::plonk::halo2::bn254::Halo2LS;
using LogDerivativeHalo2LS = c::zk::plonk::halo2::bn254::LogDerivativeHalo2LS;

template <typename PCS, typename LS>
using ProverImpl = c::zk::plonk::halo2::KZGFamilyProverImpl<PCS, LS>;

struct Param {
  uint8_t pcs_type;
  uint8_t ls_type;
  uint8_t transcript_type;

  Param(uint8_t pcs_type, uint8_t ls_type, uint8_t transcript_type)
      : pcs_type(pcs_type),
        ls_type(ls_type),
        transcript_type(transcript_type) {}
};

class ProverTest : public testing::TestWithParam<Param> {
 public:
  void SetUp() override {
    Param param = GetParam();

    k_ = 5;
    s_ = math::bn254::Fr(2);
    const tachyon_bn254_fr& c_s = c::base::c_cast(s_);
    prover_ = tachyon_halo2_bn254_prover_create_from_unsafe_setup(
        param.pcs_type, param.ls_type, param.transcript_type, k_, &c_s);
  }

  void TearDown() override { tachyon_halo2_bn254_prover_destroy(prover_); }

 protected:
  tachyon_halo2_bn254_prover* prover_ = nullptr;
  uint32_t k_;
  math::bn254::Fr s_;
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    ProverTest, ProverTest,
    testing::Values(
        Param(TACHYON_HALO2_GWC_PCS, TACHYON_HALO2_HALO2_LS,
              TACHYON_HALO2_BLAKE2B_TRANSCRIPT),
        Param(TACHYON_HALO2_GWC_PCS, TACHYON_HALO2_HALO2_LS,
              TACHYON_HALO2_POSEIDON_TRANSCRIPT),
        Param(TACHYON_HALO2_GWC_PCS, TACHYON_HALO2_HALO2_LS,
              TACHYON_HALO2_SHA256_TRANSCRIPT),
        Param(TACHYON_HALO2_GWC_PCS, TACHYON_HALO2_LOG_DERIVATIVE_HALO2_LS,
              TACHYON_HALO2_BLAKE2B_TRANSCRIPT),
        Param(TACHYON_HALO2_GWC_PCS, TACHYON_HALO2_LOG_DERIVATIVE_HALO2_LS,
              TACHYON_HALO2_POSEIDON_TRANSCRIPT),
        Param(TACHYON_HALO2_GWC_PCS, TACHYON_HALO2_LOG_DERIVATIVE_HALO2_LS,
              TACHYON_HALO2_SHA256_TRANSCRIPT),
        Param(TACHYON_HALO2_SHPLONK_PCS, TACHYON_HALO2_HALO2_LS,
              TACHYON_HALO2_BLAKE2B_TRANSCRIPT),
        Param(TACHYON_HALO2_SHPLONK_PCS, TACHYON_HALO2_HALO2_LS,
              TACHYON_HALO2_POSEIDON_TRANSCRIPT),
        Param(TACHYON_HALO2_SHPLONK_PCS, TACHYON_HALO2_HALO2_LS,
              TACHYON_HALO2_SHA256_TRANSCRIPT),
        Param(TACHYON_HALO2_SHPLONK_PCS, TACHYON_HALO2_LOG_DERIVATIVE_HALO2_LS,
              TACHYON_HALO2_BLAKE2B_TRANSCRIPT),
        Param(TACHYON_HALO2_SHPLONK_PCS, TACHYON_HALO2_LOG_DERIVATIVE_HALO2_LS,
              TACHYON_HALO2_POSEIDON_TRANSCRIPT),
        Param(TACHYON_HALO2_SHPLONK_PCS, TACHYON_HALO2_LOG_DERIVATIVE_HALO2_LS,
              TACHYON_HALO2_SHA256_TRANSCRIPT)));

TEST_P(ProverTest, Constructor) {
  Param param = GetParam();

  tachyon_halo2_bn254_prover* prover_from_halo2_parmas =
      tachyon_halo2_bn254_prover_create_from_params(
          param.pcs_type, param.ls_type, param.transcript_type, k_,
          reinterpret_cast<const uint8_t*>(
              c::zk::plonk::halo2::bn254::kExpectedHalo2Params),
          std::size(c::zk::plonk::halo2::bn254::kExpectedHalo2Params));

  const tachyon_bn254_g2_affine* expected_s_g2 =
      tachyon_halo2_bn254_prover_get_s_g2(prover_);
  const tachyon_bn254_g2_affine* s_g2 =
      tachyon_halo2_bn254_prover_get_s_g2(prover_from_halo2_parmas);
  ASSERT_TRUE(tachyon_bn254_g2_affine_eq(expected_s_g2, s_g2));
  tachyon_halo2_bn254_prover_destroy(prover_from_halo2_parmas);
}

TEST_P(ProverTest, Getters) {
  EXPECT_EQ(tachyon_halo2_bn254_prover_get_k(prover_), k_);
  EXPECT_EQ(tachyon_halo2_bn254_prover_get_n(prover_), size_t{1} << k_);
  tachyon_bn254_blinder* blinder =
      tachyon_halo2_bn254_prover_get_blinder(prover_);
  const tachyon_bn254_univariate_evaluation_domain* domain =
      tachyon_halo2_bn254_prover_get_domain(prover_);
  switch (static_cast<PCSType>(prover_->pcs_type)) {
    case PCSType::kGWC: {
      EXPECT_EQ(blinder,
                &c::base::c_cast(
                    reinterpret_cast<zk::ProverBase<GWCPCS>*>(prover_->extra)
                        ->blinder()));
      EXPECT_EQ(
          domain,
          c::base::c_cast(
              reinterpret_cast<zk::Entity<GWCPCS>*>(prover_->extra)->domain()));
      break;
    }
    case PCSType::kSHPlonk: {
      EXPECT_EQ(blinder,
                &c::base::c_cast(reinterpret_cast<zk::ProverBase<SHPlonkPCS>*>(
                                     prover_->extra)
                                     ->blinder()));
      EXPECT_EQ(domain,
                c::base::c_cast(
                    reinterpret_cast<zk::Entity<SHPlonkPCS>*>(prover_->extra)
                        ->domain()));
      break;
    }
  }

  // NOTE(dongchangYoo): |expected_s_g2| can be generated by doubling
  // g2-generator since |s| equals to 2.
  tachyon_bn254_g2_affine expected_gen = tachyon_bn254_g2_affine_generator();
  tachyon_bn254_g2_jacobian expected_s_g2_jacob =
      tachyon_bn254_g2_affine_dbl(&expected_gen);
  const tachyon_bn254_g2_affine* s_g2_affine =
      tachyon_halo2_bn254_prover_get_s_g2(prover_);
  EXPECT_EQ(c::base::native_cast(*s_g2_affine),
            c::base::native_cast(expected_s_g2_jacob).ToAffine());
}

TEST_P(ProverTest, Commit) {
  using Poly =
      math::UnivariateDensePolynomial<math::bn254::Fr, c::math::kMaxDegree>;
  Poly poly = Poly::Random(5);

  tachyon_bn254_g1_projective* point =
      tachyon_halo2_bn254_prover_commit(prover_, c::base::c_cast(&poly));

  math::bn254::G1ProjectivePoint expected;
  switch (static_cast<PCSType>(prover_->pcs_type)) {
    case PCSType::kGWC: {
      switch (static_cast<LSType>(prover_->ls_type)) {
        case LSType::kHalo2: {
          expected =
              reinterpret_cast<ProverImpl<GWCPCS, Halo2LS>*>(prover_->extra)
                  ->Commit(poly)
                  .ToProjective();
          break;
        }
        case LSType::kLogDerivativeHalo2: {
          expected =
              reinterpret_cast<ProverImpl<GWCPCS, LogDerivativeHalo2LS>*>(
                  prover_->extra)
                  ->Commit(poly)
                  .ToProjective();
          break;
        }
      }
      break;
    }
    case PCSType::kSHPlonk: {
      switch (static_cast<LSType>(prover_->ls_type)) {
        case LSType::kHalo2: {
          expected =
              reinterpret_cast<ProverImpl<SHPlonkPCS, Halo2LS>*>(prover_->extra)
                  ->Commit(poly)
                  .ToProjective();
          break;
        }
        case LSType::kLogDerivativeHalo2: {
          expected =
              reinterpret_cast<ProverImpl<SHPlonkPCS, LogDerivativeHalo2LS>*>(
                  prover_->extra)
                  ->Commit(poly)
                  .ToProjective();
          break;
        }
      }
      break;
    }
  }

  EXPECT_EQ(c::base::native_cast(*point), expected);
}

TEST_P(ProverTest, CommitLagrange) {
  using Evals =
      math::UnivariateEvaluations<math::bn254::Fr, c::math::kMaxDegree>;
  Evals evals = Evals::Random(5);

  tachyon_bn254_g1_projective* point =
      tachyon_halo2_bn254_prover_commit_lagrange(prover_,
                                                 c::base::c_cast(&evals));

  math::bn254::G1ProjectivePoint expected;
  switch (static_cast<PCSType>(prover_->pcs_type)) {
    case PCSType::kGWC: {
      switch (static_cast<LSType>(prover_->ls_type)) {
        case LSType::kHalo2: {
          expected =
              reinterpret_cast<ProverImpl<GWCPCS, Halo2LS>*>(prover_->extra)
                  ->Commit(evals)
                  .ToProjective();
          break;
        }
        case LSType::kLogDerivativeHalo2: {
          expected =
              reinterpret_cast<ProverImpl<GWCPCS, LogDerivativeHalo2LS>*>(
                  prover_->extra)
                  ->Commit(evals)
                  .ToProjective();
          break;
        }
      }
      break;
    }
    case PCSType::kSHPlonk: {
      switch (static_cast<LSType>(prover_->ls_type)) {
        case LSType::kHalo2: {
          expected =
              reinterpret_cast<ProverImpl<SHPlonkPCS, Halo2LS>*>(prover_->extra)
                  ->Commit(evals)
                  .ToProjective();
          break;
        }
        case LSType::kLogDerivativeHalo2: {
          expected =
              reinterpret_cast<ProverImpl<SHPlonkPCS, LogDerivativeHalo2LS>*>(
                  prover_->extra)
                  ->Commit(evals)
                  .ToProjective();
          break;
        }
      }
      break;
    }
  }

  EXPECT_EQ(c::base::native_cast(*point), expected);
}

TEST_P(ProverTest, BatchCommit) {
  using Poly =
      math::UnivariateDensePolynomial<math::bn254::Fr, c::math::kMaxDegree>;
  std::vector<Poly> polys =
      base::CreateVector(4, []() { return Poly::Random(5); });

  tachyon_halo2_bn254_prover_batch_start(prover_, polys.size());
  for (size_t i = 0; i < polys.size(); ++i) {
    tachyon_halo2_bn254_prover_batch_commit(prover_, c::base::c_cast(&polys[i]),
                                            i);
  }

  std::vector<math::bn254::G1AffinePoint> points(polys.size());
  tachyon_halo2_bn254_prover_batch_end(
      prover_,
      const_cast<tachyon_bn254_g1_affine*>(c::base::c_cast(points.data())),
      points.size());

  for (size_t i = 0; i < points.size(); ++i) {
    const Poly& poly = polys[i];
    math::bn254::G1AffinePoint expected;
    switch (static_cast<PCSType>(prover_->pcs_type)) {
      case PCSType::kGWC: {
        switch (static_cast<LSType>(prover_->ls_type)) {
          case LSType::kHalo2: {
            expected =
                reinterpret_cast<ProverImpl<GWCPCS, Halo2LS>*>(prover_->extra)
                    ->Commit(poly);
            break;
          }
          case LSType::kLogDerivativeHalo2: {
            expected =
                reinterpret_cast<ProverImpl<GWCPCS, LogDerivativeHalo2LS>*>(
                    prover_->extra)
                    ->Commit(poly);
            break;
          }
        }
        break;
      }
      case PCSType::kSHPlonk: {
        switch (static_cast<LSType>(prover_->ls_type)) {
          case LSType::kHalo2: {
            expected = reinterpret_cast<ProverImpl<SHPlonkPCS, Halo2LS>*>(
                           prover_->extra)
                           ->Commit(poly);
            break;
          }
          case LSType::kLogDerivativeHalo2: {
            expected =
                reinterpret_cast<ProverImpl<SHPlonkPCS, LogDerivativeHalo2LS>*>(
                    prover_->extra)
                    ->Commit(poly);
            break;
          }
        }
        break;
      }
    }
    EXPECT_EQ(points[i], expected);
  }
}

TEST_P(ProverTest, BatchCommitLagrange) {
  using Evals =
      math::UnivariateEvaluations<math::bn254::Fr, c::math::kMaxDegree>;
  std::vector<Evals> evals_vec =
      base::CreateVector(4, []() { return Evals::Random(5); });

  tachyon_halo2_bn254_prover_batch_start(prover_, evals_vec.size());
  for (size_t i = 0; i < evals_vec.size(); ++i) {
    tachyon_halo2_bn254_prover_batch_commit_lagrange(
        prover_, c::base::c_cast(&evals_vec[i]), i);
  }

  std::vector<math::bn254::G1AffinePoint> points(evals_vec.size());
  tachyon_halo2_bn254_prover_batch_end(
      prover_,
      const_cast<tachyon_bn254_g1_affine*>(c::base::c_cast(points.data())),
      points.size());

  for (size_t i = 0; i < points.size(); ++i) {
    const Evals& evals = evals_vec[i];
    math::bn254::G1AffinePoint expected;
    switch (static_cast<PCSType>(prover_->pcs_type)) {
      case PCSType::kGWC: {
        switch (static_cast<LSType>(prover_->ls_type)) {
          case LSType::kHalo2: {
            expected =
                reinterpret_cast<ProverImpl<GWCPCS, Halo2LS>*>(prover_->extra)
                    ->Commit(evals);
            break;
          }
          case LSType::kLogDerivativeHalo2: {
            expected =
                reinterpret_cast<ProverImpl<GWCPCS, LogDerivativeHalo2LS>*>(
                    prover_->extra)
                    ->Commit(evals);
            break;
          }
        }
        break;
      }
      case PCSType::kSHPlonk: {
        switch (static_cast<LSType>(prover_->ls_type)) {
          case LSType::kHalo2: {
            expected = reinterpret_cast<ProverImpl<SHPlonkPCS, Halo2LS>*>(
                           prover_->extra)
                           ->Commit(evals);
            break;
          }
          case LSType::kLogDerivativeHalo2: {
            expected =
                reinterpret_cast<ProverImpl<SHPlonkPCS, LogDerivativeHalo2LS>*>(
                    prover_->extra)
                    ->Commit(evals);
            break;
          }
        }
        break;
      }
    }
    EXPECT_EQ(points[i], expected);
  }
}

TEST_P(ProverTest, SetRng) {
  std::vector<uint8_t> seed = base::CreateVector(
      crypto::XORShiftRNG::kSeedSize,
      []() { return base::Uniform(base::Range<uint8_t>()); });
  tachyon_rng* rng = tachyon_rng_create_from_seed(TACHYON_RNG_XOR_SHIFT,
                                                  seed.data(), seed.size());
  uint8_t state[crypto::XORShiftRNG::kStateSize];
  size_t state_len;
  tachyon_rng_get_state(rng, state, &state_len);
  tachyon_halo2_bn254_prover_set_rng_state(prover_, state, state_len);

  auto cpp_rng = std::make_unique<crypto::XORShiftRNG>();
  ASSERT_TRUE(cpp_rng->SetSeed(seed));
  auto cpp_generator =
      std::make_unique<RandomFieldGenerator<math::bn254::Fr>>(cpp_rng.get());

  math::bn254::Fr expected;
  switch (static_cast<PCSType>(prover_->pcs_type)) {
    case PCSType::kGWC: {
      expected = reinterpret_cast<zk::ProverBase<GWCPCS>*>(prover_->extra)
                     ->blinder()
                     .Generate();
      break;
    }
    case PCSType::kSHPlonk: {
      expected = reinterpret_cast<zk::ProverBase<SHPlonkPCS>*>(prover_->extra)
                     ->blinder()
                     .Generate();
      break;
    }
  }

  EXPECT_EQ(cpp_generator->Generate(), expected);

  tachyon_rng_destroy(rng);
}

TEST_P(ProverTest, SetTranscript) {
  uint8_t transcript_type = GetParam().transcript_type;

  tachyon_halo2_bn254_transcript_writer* transcript =
      tachyon_halo2_bn254_transcript_writer_create(transcript_type);

  size_t digest_len = 0;
  size_t state_len = 0;
  switch (static_cast<TranscriptType>(transcript_type)) {
    case TranscriptType::kBlake2b: {
      Blake2bWriter<math::bn254::G1AffinePoint>* blake2b =
          reinterpret_cast<Blake2bWriter<math::bn254::G1AffinePoint>*>(
              transcript->extra);
      digest_len = blake2b->GetDigestLen();
      state_len = blake2b->GetStateLen();
      break;
    }
    case TranscriptType::kPoseidon: {
      PoseidonWriter<math::bn254::G1AffinePoint>* poseidon =
          reinterpret_cast<PoseidonWriter<math::bn254::G1AffinePoint>*>(
              transcript->extra);
      digest_len = poseidon->GetDigestLen();
      // NOTE(chokobole): In case of Poseidon transcript,
      // |tachyon_halo2_bn254_transcript_writer_update()| touches an internal
      // member |absorbing_|, so |state_len| has to be updated after this
      break;
    }
    case TranscriptType::kSha256: {
      Sha256Writer<math::bn254::G1AffinePoint>* sha256 =
          reinterpret_cast<Sha256Writer<math::bn254::G1AffinePoint>*>(
              transcript->extra);
      digest_len = sha256->GetDigestLen();
      state_len = sha256->GetStateLen();
      break;
    }
  }

  std::vector<uint8_t> data = base::CreateVector(
      digest_len, []() { return base::Uniform(base::Range<uint8_t>()); });
  tachyon_halo2_bn254_transcript_writer_update(transcript, data.data(),
                                               data.size());

  if (static_cast<TranscriptType>(transcript_type) ==
      TranscriptType::kPoseidon) {
    PoseidonWriter<math::bn254::G1AffinePoint>* poseidon =
        reinterpret_cast<PoseidonWriter<math::bn254::G1AffinePoint>*>(
            transcript->extra);
    state_len = poseidon->GetStateLen();
  }

  std::vector<uint8_t> state(state_len);
  tachyon_halo2_bn254_transcript_writer_get_state(transcript, state.data(),
                                                  &state_len);
  tachyon_halo2_bn254_prover_set_transcript_state(prover_, state.data(),
                                                  state_len);

  math::bn254::Fr expected;
  switch (static_cast<PCSType>(prover_->pcs_type)) {
    case PCSType::kGWC: {
      expected = reinterpret_cast<zk::Entity<GWCPCS>*>(prover_->extra)
                     ->transcript()
                     ->SqueezeChallenge();
      break;
    }
    case PCSType::kSHPlonk: {
      expected = reinterpret_cast<zk::Entity<SHPlonkPCS>*>(prover_->extra)
                     ->transcript()
                     ->SqueezeChallenge();
      break;
    }
  }

  EXPECT_EQ(
      reinterpret_cast<crypto::TranscriptWriter<math::bn254::G1AffinePoint>*>(
          transcript->extra)
          ->SqueezeChallenge(),
      expected);

  tachyon_halo2_bn254_transcript_writer_destroy(transcript);
}

}  // namespace tachyon::zk::plonk::halo2::bn254
