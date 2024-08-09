// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_PROVE_H_
#define TACHYON_ZK_R1CS_GROTH16_PROVE_H_

#include <stddef.h>

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/optional.h"
#include "tachyon/base/profiler.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/zk/r1cs/constraint_system/qap_witness_map_result.h"
#include "tachyon/zk/r1cs/groth16/proof.h"
#include "tachyon/zk/r1cs/groth16/proving_key.h"

#if TACHYON_CUDA
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_gpu.h"
#endif

namespace tachyon::zk::r1cs::groth16 {

template <typename MSM, typename Bucket, typename AffinePoint, typename F>
Bucket CalculateCoeff(MSM& msm, const Bucket& initial,
                      absl::Span<const AffinePoint> query,
                      const AffinePoint& vk_param,
                      absl::Span<const F> assignments) {
  Bucket acc;
  CHECK(msm.Run(query.subspan(1), assignments, &acc))
      << "If you encounter this error with `--config cuda`, it indicates that "
         "your GPU RAM is insufficient to hold the twiddle caches created "
         "during the icicle NTT domain initialization. If you run this from "
         "the circom prover, try using the `--disable_fast_twiddles_mode` "
         "flag.";

  Bucket ret = initial + query[0];
  ret += acc;
  ret += vk_param;
  return ret;
}

template <typename Curve, typename F>
Proof<Curve> CreateProofWithAssignment(const ProvingKey<Curve>& pk, const F& r,
                                       const F& s,
                                       absl::Span<const F> h_coefficients,
                                       absl::Span<const F> instance_assignments,
                                       absl::Span<const F> witness_assignments,
                                       absl::Span<const F> full_assignments) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

  TRACE_EVENT("ProofGeneration", "Groth16::CreateProofWithAssignment");

#if TACHYON_CUDA
  using G1Bucket = typename math::VariableBaseMSMGpu<G1AffinePoint>::Bucket;
  using G2Bucket = typename math::VariableBaseMSMGpu<G2AffinePoint>::Bucket;

  gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                           gpuMemHandleTypeNone,
                           {gpuMemLocationTypeDevice, 0}};
  device::gpu::ScopedMemPool mem_pool = device::gpu::CreateMemPool(&props);

  uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
  gpuError_t error = gpuMemPoolSetAttribute(
      mem_pool.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
  CHECK_EQ(error, gpuSuccess);
  device::gpu::ScopedStream stream = device::gpu::CreateStream();

  math::VariableBaseMSMGpu<G1AffinePoint> msm_g1(mem_pool.get(), stream.get());

  device::gpu::ScopedMemPool mem_pool2 = device::gpu::CreateMemPool(&props);

  error = gpuMemPoolSetAttribute(
      mem_pool2.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
  CHECK_EQ(error, gpuSuccess);
  device::gpu::ScopedStream stream2 = device::gpu::CreateStream();

  math::VariableBaseMSMGpu<G2AffinePoint> msm_g2(mem_pool2.get(),
                                                 stream2.get());
#else
  using G1Bucket = typename math::VariableBaseMSM<G1AffinePoint>::Bucket;
  using G2Bucket = typename math::VariableBaseMSM<G2AffinePoint>::Bucket;

  math::VariableBaseMSM<G1AffinePoint> msm_g1;
  math::VariableBaseMSM<G2AffinePoint> msm_g2;
#endif

  // |witness_acc| = [Σᵢ₌ₗ₊₁..ₘ (β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / δ]₁
  G1Bucket witness_acc;
  CHECK(msm_g1.Run(pk.l_g1_query(), witness_assignments, &witness_acc));

  // |h_acc| = [(h(x) * t(x)) / δ]₁
  G1Bucket h_acc;
  if (h_coefficients.size() > pk.h_g1_query().size()) {
    absl::Span<const F> h_coefficients_subspan =
        h_coefficients.subspan(0, h_coefficients.size() - 1);
    CHECK(msm_g1.Run(pk.h_g1_query(), h_coefficients_subspan, &h_acc));
  } else {
    absl::Span<const G1AffinePoint> h_g1_query_subspan =
        pk.h_g1_query().subspan(0, h_coefficients.size());
    CHECK(msm_g1.Run(h_g1_query_subspan, h_coefficients, &h_acc));
  }

  G1Bucket ac_g1_bucket[2];

  // |r_delta_g1_bucket| = [rδ]₁
  G1Bucket r_delta_g1_bucket = math::ConvertPoint<G1Bucket>(r * pk.delta_g1());
  // |ac_g1_bucket[0]| = [A]₁ = [α + Σᵢ₌₀..ₘ (xᵢ * aᵢ(x)) + rδ]₁
  // where x is |full_assignments|.
  ac_g1_bucket[0] =
      CalculateCoeff(msm_g1, r_delta_g1_bucket, pk.a_g1_query(),
                     pk.verifying_key().alpha_g1(), full_assignments);

  // |s_delta_g2_bucket| = [sδ]₂
  G2Bucket s_delta_g2_bucket =
      math::ConvertPoint<G2Bucket>(s * pk.verifying_key().delta_g2());
  // |b_g2_bucket| = [B]₂ = [β + Σᵢ₌₀..ₘ (xᵢ * bᵢ(x)) + sδ]₂
  // where x is |full_assignments|.
  G2Bucket b_g2_bucket =
      CalculateCoeff(msm_g2, s_delta_g2_bucket, pk.b_g2_query(),
                     pk.verifying_key().beta_g2(), full_assignments);

  // |ac_g1_bucket[1]| = [As]₁
  ac_g1_bucket[1] = ac_g1_bucket[0] * s;
  // |ac_g1_bucket[1]| = [As + Br - rsδ]₁
  if (!r.IsZero()) {
    // |s_delta_g1_bucket| = [sδ]₁
    G1Bucket s_delta_g1_bucket =
        math::ConvertPoint<G1Bucket>(s * pk.delta_g1());
    // |b_g1_bucket| = [B]₁ = [β + Σᵢ₌₀..ₘ (xᵢ * bᵢ(x)) + sδ]₁
    // where x is |full_assignments|.
    G1Bucket b_g1_bucket =
        CalculateCoeff(msm_g1, s_delta_g1_bucket, pk.b_g1_query(), pk.beta_g1(),
                       full_assignments);
    ac_g1_bucket[1] += (r * b_g1_bucket);
    ac_g1_bucket[1] -= (s * r_delta_g1_bucket);
  }
  // clang-format off
  // |ac_g1_bucket[1]| = [Σᵢ₌ₗ₊₁..ₘ (β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) / δ + As + Br - rsδ]₁
  // clang-format on
  ac_g1_bucket[1] += witness_acc;
  // clang-format off
  // |ac_g1_bucket[1]| = [C]₁ = [(Σᵢ₌ₗ₊₁..ₘ (β * aᵢ(x) + α * bᵢ(x) + cᵢ(x)) + h(x)t(x)) / δ + As + Br - rsδ]₁
  // clang-format on
  ac_g1_bucket[1] += h_acc;

  G1AffinePoint ac_g1[2];
  CHECK(G1Bucket::BatchNormalize(ac_g1_bucket, &ac_g1));

  return {
      std::move(ac_g1[0]),
      b_g2_bucket.ToAffine(),
      std::move(ac_g1[1]),
  };
}

template <typename Curve, typename F>
Proof<Curve> CreateProofWithAssignmentZK(
    const ProvingKey<Curve>& pk, absl::Span<const F> h_coefficients,
    absl::Span<const F> instance_assignments,
    absl::Span<const F> witness_assignments,
    absl::Span<const F> full_assignments) {
  return CreateProofWithAssignment(pk, F::Random(), F::Random(), h_coefficients,
                                   instance_assignments, witness_assignments,
                                   full_assignments);
}

template <typename Curve, typename F>
Proof<Curve> CreateProofWithAssignmentNoZK(
    const ProvingKey<Curve>& pk, absl::Span<const F> h_coefficients,
    absl::Span<const F> instance_assignments,
    absl::Span<const F> witness_assignments,
    absl::Span<const F> full_assignments) {
  return CreateProofWithAssignment(pk, F::Zero(), F::Zero(), h_coefficients,
                                   instance_assignments, witness_assignments,
                                   full_assignments);
}

// Create a Groth16 proof using randomness |r| and |s| and the provided
// R1CS-to-QAP reduction.
template <size_t MaxDegree, typename QAP, typename F, typename Curve>
Proof<Curve> CreateProofWithReduction(const Circuit<F>& circuit,
                                      const ProvingKey<Curve>& pk, const F& r,
                                      const F& s) {
  using Domain = math::UnivariateEvaluationDomain<F, MaxDegree>;

  TRACE_EVENT("ProofGeneration", "Groth16::CreateProofWithReduction");

  ConstraintSystem<F> cs;
  cs.set_optimization_goal(OptimizationGoal::kConstraints);

  circuit.Synthesize(cs);

  cs.Finalize();

  // TODO(chokobole): Apply |IcicleNTT|. It is tricky because we can't get the
  // domain size before calling |CreateProofWithReduction()|.
  std::unique_ptr<Domain> domain =
      Domain::Create(cs.num_constraints() + cs.num_instance_variables());

  QAPWitnessMapResult<F> result = QAP::WitnessMap(domain.get(), cs);

  const std::vector<F>& instance_assignments = cs.instance_assignments();
  const std::vector<F>& witness_assignments = cs.witness_assignments();
  return CreateProofWithAssignment(
      pk, r, s, absl::MakeConstSpan(result.h),
      absl::MakeConstSpan(instance_assignments).subspan(1),
      absl::MakeConstSpan(witness_assignments),
      absl::MakeConstSpan(result.full_assignments).subspan(1));
}

// Create a Groth16 proof that is zero-knowledge using the provided
// R1CS-to-QAP reduction.
template <size_t MaxDegree, typename QAP, typename F, typename Curve>
Proof<Curve> CreateProofWithReductionZK(const Circuit<F>& circuit,
                                        const ProvingKey<Curve>& pk) {
  return CreateProofWithReduction<MaxDegree, QAP>(circuit, pk, F::Random(),
                                                  F::Random());
}

// Create a Groth16 proof that isn't zero-knowledge using the provided
// R1CS-to-QAP reduction.
template <size_t MaxDegree, typename QAP, typename F, typename Curve>
Proof<Curve> CreateProofWithReductionNoZK(const Circuit<F>& circuit,
                                          const ProvingKey<Curve>& pk) {
  return CreateProofWithReduction<MaxDegree, QAP>(circuit, pk, F::Zero(),
                                                  F::Zero());
}

// Given a Groth16 proof, returns a fresh proof of the same statement. For a
// proof π of a statement S, the output of the non-deterministic procedure
// |ReRandomizeProof()| is statistically indistinguishable from a fresh
// honest proof of S. For more info, see theorem 3 of
// `https://eprint.iacr.org/2020/811.
template <typename Curve>
Proof<Curve> ReRandomizeProof(const VerifyingKey<Curve>& vk,
                              const Proof<Curve>& proof) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G1JacobianPoint = typename Curve::G1Curve::JacobianPoint;
  using G2JacobianPoint = typename Curve::G2Curve::JacobianPoint;
  using F = typename G1AffinePoint::ScalarField;

  TRACE_EVENT("ProofGeneration", "Groth16::ReRandomizeProof");

  struct Randoms {
    F r1;
    F r2;

    bool AreZeroes() const { return r1.IsZero() && r2.IsZero(); }

    void Sample() {
      r1 = F::Random();
      r2 = F::Random();
    }
  };

  Randoms randoms;
  while (randoms.AreZeroes()) {
    randoms.Sample();
  }

  // See figure 1 in the paper referenced above:
  //   A' = (1 / r₁)A
  //   B' = r₁B + r₁r₂(δG₂)
  //   C' = C + r₂A
  G1JacobianPoint ac_jacobian[2];
  F inv = unwrap(randoms.r1.Inverse());
  ac_jacobian[0] = proof.a() * inv;

  ac_jacobian[1] = proof.a() * randoms.r2;
  ac_jacobian[1] += proof.c();

  G1AffinePoint ac[2];
  CHECK(G1JacobianPoint::BatchNormalize(ac_jacobian, &ac));

  G2JacobianPoint b = vk.delta_g2() * randoms.r2;
  b += proof.b();
  b *= randoms.r1;

  return {
      std::move(ac[0]),
      b.ToAffine(),
      std::move(ac[1]),
  };
}

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_PROVE_H_
