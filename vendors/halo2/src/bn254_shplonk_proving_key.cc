#include "vendors/halo2/include/bn254_shplonk_proving_key.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "vendors/halo2/src/bn254.rs.h"
#include "vendors/halo2/src/shplonk_proving_key_impl.h"

namespace tachyon::halo2_api::bn254 {

namespace {

template <typename R, typename T>
rust::Vec<R> ConvertCppVecToRustVec(const std::vector<T>& vec) {
  rust::Vec<R> ret;
  ret.reserve(vec.size());
  for (const T& phase : vec) {
    ret.push_back(phase.value());
  }
  return ret;
}

template <typename R, typename T>
rust::Slice<const R> ConvertCppVecToRustSlice(const std::vector<T>& phases) {
  return {reinterpret_cast<const R*>(phases.data()), phases.size()};
}

}  // namespace

constexpr size_t kMaxDegree = (size_t{1} << 5) - 1;
constexpr size_t kMaxExtendedDegree = (size_t{1} << 7) - 1;

using PCS =
    zk::SHPlonkExtension<math::bn254::BN254Curve, kMaxDegree,
                         kMaxExtendedDegree, math::bn254::G1AffinePoint>;

class SHPlonkProvingKey::Impl : public ProvingKeyImpl<PCS> {
 public:
  explicit Impl(rust::Slice<const uint8_t> bytes)
      : ProvingKeyImpl<PCS>(bytes) {}
};

SHPlonkProvingKey::SHPlonkProvingKey(rust::Slice<const uint8_t> pk_bytes)
    : impl_(new Impl(pk_bytes)) {}

rust::Slice<const uint8_t> SHPlonkProvingKey::advice_column_phases() const {
  return ConvertCppVecToRustSlice<uint8_t>(impl_->GetAdviceColumnPhases());
}

size_t SHPlonkProvingKey::blinding_factors() const {
  return impl_->ComputeBlindingFactors();
}

rust::Slice<const uint8_t> SHPlonkProvingKey::challenge_phases() const {
  return ConvertCppVecToRustSlice<uint8_t>(impl_->GetChallengePhases());
}

rust::Vec<size_t> SHPlonkProvingKey::constants() const {
  const std::vector<zk::FixedColumnKey>& constants = impl_->GetConstants();
  rust::Vec<size_t> ret;
  ret.reserve(constants.size());
  for (const zk::FixedColumnKey& column : constants) {
    ret.push_back(column.index());
  }
  return ret;
}

size_t SHPlonkProvingKey::num_advice_columns() const {
  return impl_->GetNumAdviceColumns();
}

size_t SHPlonkProvingKey::num_challenges() const {
  return impl_->GetNumChallenges();
}

size_t SHPlonkProvingKey::num_instance_columns() const {
  return impl_->GetNumInstanceColumns();
}

rust::Vec<uint8_t> SHPlonkProvingKey::phases() const {
  return ConvertCppVecToRustVec<uint8_t>(impl_->GetPhases());
}

std::unique_ptr<SHPlonkProvingKey> new_proving_key(
    rust::Slice<const uint8_t> pk_bytes) {
  return std::make_unique<SHPlonkProvingKey>(pk_bytes);
}

}  // namespace tachyon::halo2_api::bn254
