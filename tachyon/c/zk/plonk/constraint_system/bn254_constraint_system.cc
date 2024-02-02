#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system.h"

#include <vector>

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"

using namespace tachyon;

using CS = zk::plonk::ConstraintSystem<math::bn254::Fr>;

uint32_t tachyon_bn254_plonk_constraint_system_compute_blinding_factors(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return reinterpret_cast<const CS*>(cs)->ComputeBlindingFactors();
}

void tachyon_bn254_plonk_constraint_system_get_advice_column_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len) {
  const std::vector<zk::plonk::Phase>& cpp_phases =
      reinterpret_cast<const CS*>(cs)->advice_column_phases();
  *phases_len = cpp_phases.size();
  if (phases == nullptr) return;
  for (size_t i = 0; i < cpp_phases.size(); ++i) {
    phases[i].value = cpp_phases[i].value();
  }
}

void tachyon_bn254_plonk_constraint_system_get_challenge_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len) {
  const std::vector<zk::plonk::Phase>& cpp_phases =
      reinterpret_cast<const CS*>(cs)->challenge_phases();
  *phases_len = cpp_phases.size();
  if (phases == nullptr) return;
  for (size_t i = 0; i < cpp_phases.size(); ++i) {
    phases[i].value = cpp_phases[i].value();
  }
}

void tachyon_bn254_plonk_constraint_system_get_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len) {
  std::vector<zk::plonk::Phase> cpp_phases =
      reinterpret_cast<const CS*>(cs)->GetPhases();
  *phases_len = cpp_phases.size();
  if (phases == nullptr) return;
  for (size_t i = 0; i < cpp_phases.size(); ++i) {
    phases[i].value = cpp_phases[i].value();
  }
}

size_t tachyon_bn254_plonk_constraint_system_get_num_fixed_columns(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return reinterpret_cast<const CS*>(cs)->num_fixed_columns();
}

size_t tachyon_bn254_plonk_constraint_system_get_num_instance_columns(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return reinterpret_cast<const CS*>(cs)->num_instance_columns();
}

size_t tachyon_bn254_plonk_constraint_system_get_num_advice_columns(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return reinterpret_cast<const CS*>(cs)->num_advice_columns();
}

size_t tachyon_bn254_plonk_constraint_system_get_num_challenges(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return reinterpret_cast<const CS*>(cs)->num_challenges();
}

void tachyon_bn254_plonk_constraint_system_get_constants(
    const tachyon_bn254_plonk_constraint_system* cs,
    tachyon_fixed_column_key* constants, size_t* constants_len) {
  const std::vector<zk::plonk::FixedColumnKey>& cpp_constants =
      reinterpret_cast<const CS*>(cs)->constants();
  *constants_len = cpp_constants.size();
  if (constants == nullptr) return;
  for (size_t i = 0; i < cpp_constants.size(); ++i) {
    constants[i].index = cpp_constants[i].index();
  }
}
