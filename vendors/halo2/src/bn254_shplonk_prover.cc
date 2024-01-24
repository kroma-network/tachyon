#include "vendors/halo2/include/bn254_shplonk_prover.h"

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/rs/base/container_util.h"
#include "vendors/halo2/src/bn254.rs.h"
#include "vendors/halo2/src/bn254_shplonk_prover_impl.h"
#include "vendors/halo2/src/bn254_shplonk_proving_key_impl.h"

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

void SHPlonkProver::set_rng(rust::Slice<const uint8_t> state) {
  base::Buffer buffer(const_cast<uint8_t*>(state.data()), state.size());
  uint32_t x, y, z, w;
  CHECK(buffer.Read32LE(&x));
  CHECK(buffer.Read32LE(&y));
  CHECK(buffer.Read32LE(&z));
  CHECK(buffer.Read32LE(&w));
  impl_->SetRng(std::make_unique<crypto::XORShiftRNG>(
      crypto::XORShiftRNG::FromState(x, y, z, w)));
}

void SHPlonkProver::set_transcript(rust::Slice<const uint8_t> state) {
  base::Uint8VectorBuffer write_buf;
  std::unique_ptr<zk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>> writer =
      std::make_unique<zk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>(
          std::move(write_buf));
  writer->SetState(rs::ConvertRustSliceToCppSpan<const uint8_t>(state));
  impl_->SetTranscript(std::move(writer));
}

void SHPlonkProver::set_extended_domain(const SHPlonkProvingKey& pk) {
  impl_->SetExtendedDomain(pk.impl()->GetConstraintSystem());
}

std::unique_ptr<SHPlonkProver> new_shplonk_prover(uint32_t k, const Fr& s) {
  return std::make_unique<SHPlonkProver>(k, s);
}

}  // namespace tachyon::halo2_api::bn254
