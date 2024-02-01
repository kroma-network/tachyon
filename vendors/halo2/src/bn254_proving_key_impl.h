#ifndef VENDORS_HALO2_SRC_BN254_PROVING_KEY_IMPL_H_
#define VENDORS_HALO2_SRC_BN254_PROVING_KEY_IMPL_H_

#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

namespace tachyon::halo2_api::bn254 {

class ProvingKeyImpl {
 public:
  explicit ProvingKeyImpl(rust::Slice<const uint8_t> pk_bytes)
      : pk_(tachyon_bn254_plonk_proving_key_create_from_state(
            pk_bytes.data(), pk_bytes.size())) {}
  ProvingKeyImpl(const ProvingKeyImpl& other) = delete;
  ProvingKeyImpl& operator=(const ProvingKeyImpl& other) = delete;
  ~ProvingKeyImpl() { tachyon_bn254_plonk_proving_key_destroy(pk_); }

  tachyon_bn254_plonk_proving_key* pk() { return pk_; }
  const tachyon_bn254_plonk_proving_key* pk() const { return pk_; }

  rust::Vec<uint8_t> GetAdviceColumnPhases() const {
    return DoGetPhases(
        GetConstraintSystem(),
        &tachyon_bn254_plonk_constraint_system_get_advice_column_phases);
  }

  uint32_t ComputeBlindingFactors() const {
    return tachyon_bn254_plonk_constraint_system_compute_blinding_factors(
        GetConstraintSystem());
  }

  rust::Vec<uint8_t> GetChallengePhases() const {
    return DoGetPhases(
        GetConstraintSystem(),
        &tachyon_bn254_plonk_constraint_system_get_challenge_phases);
  }

  rust::Vec<size_t> GetConstants() const {
    return GetFixedColumns(
        GetConstraintSystem(),
        &tachyon_bn254_plonk_constraint_system_get_constants);
  }

  size_t GetNumAdviceColumns() const {
    return tachyon_bn254_plonk_constraint_system_get_num_advice_columns(
        GetConstraintSystem());
  }

  size_t GetNumChallenges() const {
    return tachyon_bn254_plonk_constraint_system_get_num_challenges(
        GetConstraintSystem());
  }

  size_t GetNumInstanceColumns() const {
    return tachyon_bn254_plonk_constraint_system_get_num_instance_columns(
        GetConstraintSystem());
  }

  rust::Vec<uint8_t> GetPhases() const {
    return DoGetPhases(GetConstraintSystem(),
                       &tachyon_bn254_plonk_constraint_system_get_phases);
  }

 private:
  using GetPhasesAPI = void (*)(const tachyon_bn254_plonk_constraint_system*,
                                tachyon_phase*, size_t*);
  using GetFixedColumnsAPI =
      void (*)(const tachyon_bn254_plonk_constraint_system*,
               tachyon_fixed_column_key*, size_t*);

  const tachyon_bn254_plonk_verifying_key* GetVerifyingKey() const {
    return tachyon_bn254_plonk_proving_key_get_verifying_key(pk_);
  }

  const tachyon_bn254_plonk_constraint_system* GetConstraintSystem() const {
    return tachyon_bn254_plonk_verifying_key_get_constraint_system(
        GetVerifyingKey());
  }

  static rust::Vec<uint8_t> DoGetPhases(
      const tachyon_bn254_plonk_constraint_system* cs, GetPhasesAPI api) {
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

  static rust::Vec<size_t> GetFixedColumns(
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

  tachyon_bn254_plonk_proving_key* pk_;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_SRC_BN254_PROVING_KEY_IMPL_H_
