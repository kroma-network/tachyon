#include "vendors/halo2/include/bn254_proving_key.h"

#include "tachyon/rs/base/container_util.h"
#include "vendors/halo2/src/bn254.rs.h"
#include "vendors/halo2/src/bn254_proving_key_impl.h"

namespace tachyon::halo2_api::bn254 {

ProvingKey::ProvingKey(rust::Slice<const uint8_t> pk_bytes)
    : impl_(new ProvingKeyImpl(pk_bytes)) {}

rust::Slice<const uint8_t> ProvingKey::advice_column_phases() const {
  return rs::ConvertCppContainerToRustSlice<uint8_t>(
      impl_->GetAdviceColumnPhases());
}

uint32_t ProvingKey::blinding_factors() const {
  return impl_->ComputeBlindingFactors();
}

rust::Slice<const uint8_t> ProvingKey::challenge_phases() const {
  return rs::ConvertCppContainerToRustSlice<uint8_t>(
      impl_->GetChallengePhases());
}

rust::Vec<size_t> ProvingKey::constants() const {
  const std::vector<zk::FixedColumnKey>& constants = impl_->GetConstants();
  rust::Vec<size_t> ret;
  ret.reserve(constants.size());
  for (const zk::FixedColumnKey& column : constants) {
    ret.push_back(column.index());
  }
  return ret;
}

size_t ProvingKey::num_advice_columns() const {
  return impl_->GetNumAdviceColumns();
}

size_t ProvingKey::num_challenges() const { return impl_->GetNumChallenges(); }

size_t ProvingKey::num_instance_columns() const {
  return impl_->GetNumInstanceColumns();
}

rust::Vec<uint8_t> ProvingKey::phases() const {
  return rs::ConvertCppContainerToRustVec(
      impl_->GetPhases(), [](zk::Phase phase) { return phase.value(); });
}

std::unique_ptr<ProvingKey> new_proving_key(
    rust::Slice<const uint8_t> pk_bytes) {
  return std::make_unique<ProvingKey>(pk_bytes);
}

}  // namespace tachyon::halo2_api::bn254
