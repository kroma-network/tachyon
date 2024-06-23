#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system.h"

#include <vector>

#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system_type_traits.h"

using namespace tachyon;

uint32_t tachyon_bn254_plonk_constraint_system_compute_blinding_factors(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return c::base::native_cast(cs)->ComputeBlindingFactors();
}

void tachyon_bn254_plonk_constraint_system_get_advice_column_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len) {
  const std::vector<zk::plonk::Phase>& cpp_phases =
      c::base::native_cast(cs)->advice_column_phases();
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
      c::base::native_cast(cs)->challenge_phases();
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
      c::base::native_cast(cs)->GetPhases();
  *phases_len = cpp_phases.size();
  if (phases == nullptr) return;
  for (size_t i = 0; i < cpp_phases.size(); ++i) {
    phases[i].value = cpp_phases[i].value();
  }
}

size_t tachyon_bn254_plonk_constraint_system_get_num_fixed_columns(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return c::base::native_cast(cs)->num_fixed_columns();
}

size_t tachyon_bn254_plonk_constraint_system_get_num_instance_columns(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return c::base::native_cast(cs)->num_instance_columns();
}

size_t tachyon_bn254_plonk_constraint_system_get_num_advice_columns(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return c::base::native_cast(cs)->num_advice_columns();
}

size_t tachyon_bn254_plonk_constraint_system_get_num_challenges(
    const tachyon_bn254_plonk_constraint_system* cs) {
  return c::base::native_cast(cs)->num_challenges();
}

void tachyon_bn254_plonk_constraint_system_get_constants(
    const tachyon_bn254_plonk_constraint_system* cs,
    tachyon_fixed_column_key* constants, size_t* constants_len) {
  const std::vector<zk::plonk::FixedColumnKey>& cpp_constants =
      c::base::native_cast(cs)->constants();
  *constants_len = cpp_constants.size();
  if (constants == nullptr) return;
  for (size_t i = 0; i < cpp_constants.size(); ++i) {
    constants[i].index = cpp_constants[i].index();
  }
}
