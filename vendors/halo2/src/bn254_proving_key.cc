#include "vendors/halo2/include/bn254_proving_key.h"

#include "vendors/halo2/src/bn254.rs.h"

namespace tachyon::halo2_api::bn254 {

namespace {

using GetPhasesAPI = void (*)(const tachyon_bn254_plonk_constraint_system*,
                              tachyon_phase*, size_t*);
using GetFixedColumnsAPI =
    void (*)(const tachyon_bn254_plonk_constraint_system*,
             tachyon_fixed_column_key*, size_t*);

rust::Vec<uint8_t> DoGetPhases(const tachyon_bn254_plonk_constraint_system* cs,
                               GetPhasesAPI api) {
  static_assert(sizeof(uint8_t) == sizeof(tachyon_phase));
  rust::Vec<uint8_t> phases;
  size_t phases_len;
  api(cs, nullptr, &phases_len);
  phases.reserve(phases_len);
  for (size_t i = 0; i < phases_len; ++i) {
    phases.push_back(0);
  }
  api(cs, reinterpret_cast<tachyon_phase*>(phases.data()), &phases_len);
  return phases;
}

rust::Vec<size_t> GetFixedColumns(
    const tachyon_bn254_plonk_constraint_system* cs, GetFixedColumnsAPI api) {
  static_assert(sizeof(size_t) == sizeof(tachyon_fixed_column_key));
  rust::Vec<size_t> fixed_columns;
  size_t fixed_columns_len;
  api(cs, nullptr, &fixed_columns_len);
  fixed_columns.reserve(fixed_columns_len);
  for (size_t i = 0; i < fixed_columns_len; ++i) {
    fixed_columns.push_back(0);
  }
  api(cs, reinterpret_cast<tachyon_fixed_column_key*>(fixed_columns.data()),
      &fixed_columns_len);
  return fixed_columns;
}

}  // namespace

ProvingKey::ProvingKey(rust::Slice<const uint8_t> pk_bytes)
    : pk_(tachyon_bn254_plonk_proving_key_create_from_state(
          TACHYON_HALO2_HALO2_LS, pk_bytes.data(), pk_bytes.size())) {}

ProvingKey::~ProvingKey() { tachyon_bn254_plonk_proving_key_destroy(pk_); }

rust::Vec<uint8_t> ProvingKey::advice_column_phases() const {
  return DoGetPhases(
      GetConstraintSystem(),
      &tachyon_bn254_plonk_constraint_system_get_advice_column_phases);
}

uint32_t ProvingKey::blinding_factors() const {
  return tachyon_bn254_plonk_constraint_system_compute_blinding_factors(
      GetConstraintSystem());
}

rust::Vec<uint8_t> ProvingKey::challenge_phases() const {
  return DoGetPhases(
      GetConstraintSystem(),
      &tachyon_bn254_plonk_constraint_system_get_challenge_phases);
}

rust::Vec<size_t> ProvingKey::constants() const {
  return GetFixedColumns(GetConstraintSystem(),
                         &tachyon_bn254_plonk_constraint_system_get_constants);
}

size_t ProvingKey::num_advice_columns() const {
  return tachyon_bn254_plonk_constraint_system_get_num_advice_columns(
      GetConstraintSystem());
}

size_t ProvingKey::num_challenges() const {
  return tachyon_bn254_plonk_constraint_system_get_num_challenges(
      GetConstraintSystem());
}

size_t ProvingKey::num_instance_columns() const {
  return tachyon_bn254_plonk_constraint_system_get_num_instance_columns(
      GetConstraintSystem());
}

rust::Vec<uint8_t> ProvingKey::phases() const {
  return DoGetPhases(GetConstraintSystem(),
                     &tachyon_bn254_plonk_constraint_system_get_phases);
}

const tachyon_bn254_plonk_verifying_key* ProvingKey::GetVerifyingKey() const {
  return tachyon_bn254_plonk_proving_key_get_verifying_key(pk_);
}

const tachyon_bn254_plonk_constraint_system* ProvingKey::GetConstraintSystem()
    const {
  return tachyon_bn254_plonk_verifying_key_get_constraint_system(
      GetVerifyingKey());
}

std::unique_ptr<ProvingKey> new_proving_key(
    rust::Slice<const uint8_t> pk_bytes) {
  return std::make_unique<ProvingKey>(pk_bytes);
}

}  // namespace tachyon::halo2_api::bn254
