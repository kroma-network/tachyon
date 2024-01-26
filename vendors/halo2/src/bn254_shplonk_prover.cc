#include "vendors/halo2/include/bn254_shplonk_prover.h"

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/rs/base/container_util.h"
#include "tachyon/rs/base/rust_vec_copyable.h"
#include "vendors/halo2/include/bn254_evals.h"
#include "vendors/halo2/include/bn254_poly.h"
#include "vendors/halo2/src/bn254.rs.h"
#include "vendors/halo2/src/bn254_evals_impl.h"
#include "vendors/halo2/src/bn254_poly_impl.h"
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

std::vector<math::bn254::Fr> ReadFrsToRemove(uint8_t* vec_ptr, size_t i) {
  size_t num_bytes = base::EstimateSize(rs::RustVec());
  base::Buffer buffer(&vec_ptr[num_bytes * i], num_bytes);
  rs::RustVec vec;
  CHECK(buffer.Read(&vec));
  return vec.ToVec<math::bn254::Fr>();
}

PCS::Evals ReadEvalsToRemove(uint8_t* vec_ptr, size_t i) {
  return PCS::Evals(ReadFrsToRemove(vec_ptr, i));
}

PCS::Evals ReadEvals(uint8_t* vec_ptr, size_t i) {
  size_t num_bytes = sizeof(uintptr_t);
  base::Buffer buffer(&vec_ptr[num_bytes * i], num_bytes);
  uintptr_t ptr;
  CHECK(buffer.Read(&ptr));
  Evals* evals = reinterpret_cast<Evals*>(ptr);
  return std::move(*evals->impl()).TakeEvals();
}

PCS::Poly ReadPoly(uint8_t* vec_ptr, size_t i) {
  size_t num_bytes = sizeof(uintptr_t);
  base::Buffer buffer(&vec_ptr[num_bytes * i], num_bytes);
  uintptr_t ptr;
  CHECK(buffer.Read(&ptr));
  Poly* poly = reinterpret_cast<Poly*>(ptr);
  return std::move(*poly->impl()).TakePoly();
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

std::unique_ptr<Evals> SHPlonkProver::empty_evals() const {
  const PCS::Domain* domain = impl_->prover().domain();
  PCS::Evals evals = domain->Empty<PCS::Evals>();
  std::unique_ptr<Evals> ret(new Evals());
  PCS::Evals& impl = reinterpret_cast<PCS::Evals&>(ret->impl()->evals());
  impl = std::move(evals);
  return ret;
}

std::unique_ptr<Poly> SHPlonkProver::ifft(const Evals& evals) const {
  const PCS::Domain* domain = impl_->prover().domain();
  PCS::Poly poly =
      domain->IFFT(reinterpret_cast<const PCS::Evals&>(evals.impl()->evals()));
  std::unique_ptr<Poly> ret(new Poly());
  PCS::Poly& impl = reinterpret_cast<PCS::Poly&>(ret->impl()->poly());
  impl = std::move(poly);
  // NOTE(chokobole): The zero degrees might be removed. This is not compatible
  // with rust halo2.
  impl.coefficients().coefficients().resize(domain->size());
  return ret;
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

void SHPlonkProver::create_proof(const SHPlonkProvingKey& key,
                                 rust::Vec<InstanceSingle> instance_singles,
                                 rust::Vec<AdviceSingle> advice_singles,
                                 rust::Vec<Fr> challenges) {
  const zk::ProvingKey<PCS>& cpp_key = key.impl()->key();
  impl_->SetBlindingFactors(
      cpp_key.verifying_key().constraint_system().ComputeBlindingFactors());

  size_t num_circuits = instance_singles.size();
  CHECK_EQ(num_circuits, advice_singles.size())
      << "size of |instance_singles| and |advice_singles| don't match";

  std::vector<std::vector<PCS::Evals>> advice_columns_vec;
  advice_columns_vec.resize(num_circuits);
  std::vector<std::vector<math::bn254::Fr>> advice_blinds_vec;
  advice_blinds_vec.resize(num_circuits);

  std::vector<math::bn254::Fr> cpp_challenges =
      base::Map(challenges, [](const Fr& fr) {
        return reinterpret_cast<const math::bn254::Fr&>(fr);
      });

  std::vector<std::vector<PCS::Evals>> instance_columns_vec;
  instance_columns_vec.resize(num_circuits);
  std::vector<std::vector<PCS::Poly>> instance_polys_vec;
  instance_polys_vec.resize(num_circuits);

  // TODO(chokobole): We shouldn't copy values here in the next iteration.
  size_t num_bytes = base::EstimateSize(rs::RustVec());
  for (size_t i = 0; i < num_circuits; ++i) {
    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(advice_singles.data());
    base::Buffer buffer(&buffer_ptr[num_bytes * 2 * i], num_bytes * 2);
    rs::RustVec vec;

    CHECK(buffer.Read(&vec));
    size_t num_advice_columns = vec.length;
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(vec.ptr);
    advice_columns_vec[i] = base::CreateVector(
        num_advice_columns,
        [vec_ptr](size_t j) { return ReadEvalsToRemove(vec_ptr, j); });

    CHECK(buffer.Read(&vec));
    vec_ptr = reinterpret_cast<uint8_t*>(vec.ptr);
    advice_blinds_vec[i] = vec.ToVec<math::bn254::Fr>();

    buffer_ptr = reinterpret_cast<uint8_t*>(instance_singles.data());
    buffer = base::Buffer(&buffer_ptr[num_bytes * 2 * i], num_bytes * 2);

    CHECK(buffer.Read(&vec));
    size_t num_instance_columns = vec.length;
    vec_ptr = reinterpret_cast<uint8_t*>(vec.ptr);
    instance_columns_vec[i] = base::CreateVector(
        num_instance_columns,
        [vec_ptr](size_t j) { return ReadEvals(vec_ptr, j); });

    CHECK(buffer.Read(&vec));
    CHECK_EQ(num_instance_columns, vec.length)
        << "size of instance columns don't match";
    vec_ptr = reinterpret_cast<uint8_t*>(vec.ptr);
    instance_polys_vec[i] = base::CreateVector(
        num_instance_columns,
        [vec_ptr](size_t j) { return ReadPoly(vec_ptr, j); });
  }

  zk::halo2::Argument<PCS> argument(
      num_circuits, &cpp_key.fixed_columns(), &cpp_key.fixed_polys(),
      std::move(advice_columns_vec), std::move(advice_blinds_vec),
      std::move(cpp_challenges), std::move(instance_columns_vec),
      std::move(instance_polys_vec));
  impl_->CreateProof(cpp_key, argument);
}

rust::Vec<uint8_t> SHPlonkProver::finalize_transcript() {
  return rs::ConvertCppVecToRustVec(impl_->GetTranscriptOwnedBuffer());
}

std::unique_ptr<SHPlonkProver> new_shplonk_prover(uint32_t k, const Fr& s) {
  return std::make_unique<SHPlonkProver>(k, s);
}

rust::Box<Fr> SHPlonkProvingKey::transcript_repr(const SHPlonkProver& prover) {
  math::bn254::Fr* ret =
      new math::bn254::Fr(impl_->GetTranscriptRepr(prover.impl()->prover()));
  return rust::Box<Fr>::from_raw(reinterpret_cast<Fr*>(ret));
}

}  // namespace tachyon::halo2_api::bn254
