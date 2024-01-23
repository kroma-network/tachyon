#include "vendors/halo2/include/bn254_shplonk_prover.h"

#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/rs/base/container_util.h"
#include "vendors/halo2/src/bn254.rs.h"
#include "vendors/halo2/src/bn254_shplonk_prover_impl.h"

namespace tachyon::halo2_api::bn254 {

namespace {

rust::Box<G1JacobianPoint> DoCommit(
    const std::vector<math::bn254::G1AffinePoint>& cpp_bases,
    rust::Slice<const Fr> scalars) {
  math::VariableBaseMSM<math::bn254::G1AffinePoint> msm;
  math::VariableBaseMSM<math::bn254::G1AffinePoint>::Bucket bucket;
  CHECK(msm.Run(cpp_bases,
                rs::ConvertRustSliceToCppSpan<const math::bn254::Fr>(scalars),
                &bucket));
  math::bn254::G1JacobianPoint* result = new math::bn254::G1JacobianPoint;
  *result = bucket.ToJacobian();
  return rust::Box<G1JacobianPoint>::from_raw(
      reinterpret_cast<G1JacobianPoint*>(result));
}

}  // namespace

SHPlonkProver::SHPlonkProver(uint32_t k, const Fr& s)
    : impl_(new SHPlonkProverImpl(k, s)) {}

uint32_t SHPlonkProver::k() const { return static_cast<uint32_t>(impl_->K()); }

uint64_t SHPlonkProver::n() const { return static_cast<uint64_t>(impl_->N()); }

rust::Box<G1JacobianPoint> SHPlonkProver::commit(
    rust::Slice<const Fr> scalars) const {
  return DoCommit(impl_->prover().pcs().GetG1PowersOfTau(), scalars);
}

rust::Box<G1JacobianPoint> SHPlonkProver::commit_lagrange(
    rust::Slice<const Fr> scalars) const {
  return DoCommit(impl_->prover().pcs().GetG1PowersOfTauLagrange(), scalars);
}

std::unique_ptr<SHPlonkProver> new_shplonk_prover(uint32_t k, const Fr& s) {
  return std::make_unique<SHPlonkProver>(k, s);
}

}  // namespace tachyon::halo2_api::bn254
