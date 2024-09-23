#include "vendors/scroll_halo2/include/bn254_prover.h"

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"
#include "tachyon/c/zk/plonk/halo2/constants.h"
#include "tachyon/rs/base/container_util.h"
#include "tachyon/rs/base/rust_vec.h"
#include "vendors/scroll_halo2/src/bn254.rs.h"

namespace tachyon::halo2_api::bn254 {

Prover::Prover(uint8_t pcs_type, uint8_t transcript_type, uint32_t k,
               const Fr& s)
    : prover_(tachyon_halo2_bn254_prover_create_from_unsafe_setup(
          TACHYON_HALO2_SCROLL_VENDOR, pcs_type, transcript_type, k,
          reinterpret_cast<const tachyon_bn254_fr*>(&s))) {}

Prover::Prover(uint8_t pcs_type, uint8_t transcript_type, uint32_t k,
               const uint8_t* params, size_t params_len)
    : prover_(tachyon_halo2_bn254_prover_create_from_params(
          TACHYON_HALO2_SCROLL_VENDOR, pcs_type, transcript_type, k, params,
          params_len)) {}

Prover::~Prover() { tachyon_halo2_bn254_prover_destroy(prover_); }

uint32_t Prover::k() const { return tachyon_halo2_bn254_prover_get_k(prover_); }

uint64_t Prover::n() const {
  return static_cast<uint64_t>(tachyon_halo2_bn254_prover_get_n(prover_));
}

const G2AffinePoint& Prover::s_g2() const {
  return reinterpret_cast<const G2AffinePoint&>(
      *tachyon_halo2_bn254_prover_get_s_g2(prover_));
}

rust::Box<G1ProjectivePoint> Prover::commit(const Poly& poly) const {
  return rust::Box<G1ProjectivePoint>::from_raw(
      reinterpret_cast<G1ProjectivePoint*>(
          tachyon_halo2_bn254_prover_commit(prover_, poly.poly())));
}

rust::Box<G1ProjectivePoint> Prover::commit_lagrange(const Evals& evals) const {
  return rust::Box<G1ProjectivePoint>::from_raw(
      reinterpret_cast<G1ProjectivePoint*>(
          tachyon_halo2_bn254_prover_commit_lagrange(prover_, evals.evals())));
}

void Prover::batch_start(size_t len) const {
  tachyon_halo2_bn254_prover_batch_start(prover_, len);
}

void Prover::batch_commit(const Poly& poly, size_t i) const {
  tachyon_halo2_bn254_prover_batch_commit(prover_, poly.poly(), i);
}

void Prover::batch_commit_lagrange(const Evals& evals, size_t i) const {
  tachyon_halo2_bn254_prover_batch_commit_lagrange(prover_, evals.evals(), i);
}

void Prover::batch_end(rust::Slice<G1AffinePoint> points) const {
  tachyon_halo2_bn254_prover_batch_end(
      prover_,
      const_cast<tachyon_bn254_g1_affine*>(
          reinterpret_cast<const tachyon_bn254_g1_affine*>(points.data())),
      points.size());
}

std::unique_ptr<Evals> Prover::empty_evals() const {
  return std::make_unique<Evals>(
      tachyon_bn254_univariate_evaluation_domain_empty_evals(
          tachyon_halo2_bn254_prover_get_domain(prover_)));
}

std::unique_ptr<RationalEvals> Prover::empty_rational_evals() const {
  return std::make_unique<RationalEvals>(
      tachyon_bn254_univariate_evaluation_domain_empty_rational_evals(
          tachyon_halo2_bn254_prover_get_domain(prover_)));
}

std::unique_ptr<Poly> Prover::ifft(const Evals& evals) const {
  // NOTE(chokobole): Leading zero values may be removed at this point, so use
  // this function cautiously. Tachyon currently uses this safely.
  return std::make_unique<Poly>(tachyon_bn254_univariate_evaluation_domain_ifft(
      tachyon_halo2_bn254_prover_get_domain(prover_), evals.evals()));
}

void Prover::batch_evaluate(
    rust::Slice<const std::unique_ptr<RationalEvals>> rational_evals,
    rust::Slice<std::unique_ptr<Evals>> evals) const {
  for (size_t i = 0; i < rational_evals.size(); ++i) {
    evals[i] = std::make_unique<Evals>(
        tachyon_bn254_univariate_rational_evaluations_batch_evaluate(
            rational_evals[i]->evals()));
  }
}

void Prover::set_rng(uint8_t rng_type, rust::Slice<const uint8_t> state) {
  tachyon_halo2_bn254_prover_set_rng_state(prover_, rng_type, state.data(),
                                           state.size());
}

void Prover::set_transcript(rust::Slice<const uint8_t> state) {
  tachyon_halo2_bn254_prover_set_transcript_state(prover_, state.data(),
                                                  state.size());
}

void Prover::set_extended_domain(const ProvingKey& pk) {
  tachyon_halo2_bn254_prover_set_extended_domain(prover_, pk.pk());
}

void Prover::create_proof(ProvingKey& key,
                          rust::Slice<InstanceSingle> instance_singles,
                          rust::Slice<AdviceSingle> advice_singles,
                          rust::Slice<const Fr> challenges) {
  tachyon_bn254_blinder* blinder =
      tachyon_halo2_bn254_prover_get_blinder(prover_);
  const tachyon_bn254_plonk_verifying_key* vk =
      tachyon_bn254_plonk_proving_key_get_verifying_key(key.pk());
  const tachyon_bn254_plonk_constraint_system* cs =
      tachyon_bn254_plonk_verifying_key_get_constraint_system(vk);
  uint32_t blinding_factors =
      tachyon_bn254_plonk_constraint_system_compute_blinding_factors(cs);
  tachyon_halo2_bn254_blinder_set_blinding_factors(blinder, blinding_factors);

  size_t num_circuits = instance_singles.size();
  CHECK_EQ(num_circuits, advice_singles.size())
      << "size of |instance_singles| and |advice_singles| don't match";

  tachyon_halo2_bn254_argument_data* data =
      tachyon_halo2_bn254_argument_data_create(num_circuits);

  tachyon_halo2_bn254_argument_data_reserve_challenges(data, challenges.size());
  for (size_t i = 0; i < challenges.size(); ++i) {
    tachyon_halo2_bn254_argument_data_add_challenge(
        data, reinterpret_cast<const tachyon_bn254_fr*>(&challenges[i]));
  }

  size_t num_bytes = base::EstimateSize(rs::RustVec());
  for (size_t i = 0; i < num_circuits; ++i) {
    uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(advice_singles.data());
    base::Buffer buffer(&buffer_ptr[num_bytes * 2 * i], num_bytes * 2);
    rs::RustVec vec;

    CHECK(buffer.Read(&vec));
    size_t num_advice_columns = vec.length;
    uintptr_t* advice_columns_ptr = reinterpret_cast<uintptr_t*>(vec.ptr);
    tachyon_halo2_bn254_argument_data_reserve_advice_columns(
        data, i, num_advice_columns);
    for (size_t j = 0; j < num_advice_columns; ++j) {
      tachyon_halo2_bn254_argument_data_add_advice_column(
          data, i, reinterpret_cast<Evals*>(advice_columns_ptr[j])->release());
    }

    CHECK(buffer.Read(&vec));
    size_t num_blinds = vec.length;
    const tachyon_bn254_fr* blinds_ptr =
        reinterpret_cast<const tachyon_bn254_fr*>(vec.ptr);
    tachyon_halo2_bn254_argument_data_reserve_advice_blinds(data, i,
                                                            num_blinds);
    for (size_t j = 0; j < num_blinds; ++j) {
      tachyon_halo2_bn254_argument_data_add_advice_blind(data, i,
                                                         &blinds_ptr[j]);
    }

    buffer_ptr = reinterpret_cast<uint8_t*>(instance_singles.data());
    buffer = base::Buffer(&buffer_ptr[num_bytes * 2 * i], num_bytes * 2);

    CHECK(buffer.Read(&vec));
    size_t num_instance_columns = vec.length;
    uintptr_t* instance_columns_ptr = reinterpret_cast<uintptr_t*>(vec.ptr);
    tachyon_halo2_bn254_argument_data_reserve_instance_columns(
        data, i, num_instance_columns);
    for (size_t j = 0; j < num_instance_columns; ++j) {
      tachyon_halo2_bn254_argument_data_add_instance_column(
          data, i,
          reinterpret_cast<Evals*>(instance_columns_ptr[j])->release());
    }

    CHECK(buffer.Read(&vec));
    CHECK_EQ(num_instance_columns, vec.length)
        << "size of instance columns don't match";
    uintptr_t* instance_poly_ptr = reinterpret_cast<uintptr_t*>(vec.ptr);
    tachyon_halo2_bn254_argument_data_reserve_instance_polys(
        data, i, num_instance_columns);
    for (size_t j = 0; j < num_instance_columns; ++j) {
      tachyon_halo2_bn254_argument_data_add_instance_poly(
          data, i, reinterpret_cast<Poly*>(instance_poly_ptr[j])->release());
    }

    CHECK(buffer.Done());
  }

  tachyon_halo2_bn254_prover_create_proof(prover_, key.pk(), data);
  tachyon_halo2_bn254_argument_data_destroy(data);
}

rust::Vec<uint8_t> Prover::get_proof() const {
  size_t proof_len;
  tachyon_halo2_bn254_prover_get_proof(prover_, nullptr, &proof_len);
  rust::Vec<uint8_t> proof = rs::CreateDefaultVector<uint8_t>(proof_len);
  tachyon_halo2_bn254_prover_get_proof(prover_, proof.data(), &proof_len);
  return proof;
}

std::unique_ptr<Prover> new_prover(uint8_t pcs_type, uint8_t transcript_type,
                                   uint32_t k, const Fr& s) {
  return std::make_unique<Prover>(pcs_type, transcript_type, k, s);
}

std::unique_ptr<Prover> new_prover_from_params(
    uint8_t pcs_type, uint8_t transcript_type, uint32_t k,
    rust::Slice<const uint8_t> params) {
  return std::make_unique<Prover>(pcs_type, transcript_type, k, params.data(),
                                  params.size());
}

rust::Box<Fr> ProvingKey::transcript_repr(const Prover& prover) {
  tachyon_halo2_bn254_prover_set_transcript_repr(prover.prover(), pk_);
  tachyon_bn254_fr* ret = new tachyon_bn254_fr;
  tachyon_bn254_fr repr = tachyon_bn254_plonk_verifying_key_get_transcript_repr(
      tachyon_bn254_plonk_proving_key_get_verifying_key(pk_));
  memcpy(ret->limbs, repr.limbs, sizeof(uint64_t) * 4);
  return rust::Box<Fr>::from_raw(reinterpret_cast<Fr*>(ret));
}

}  // namespace tachyon::halo2_api::bn254
