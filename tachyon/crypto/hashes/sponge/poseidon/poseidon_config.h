// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_

#include <utility>

#include "absl/types/span.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/optional.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config_base.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config_entry.h"

namespace tachyon {
namespace crypto {

// ARK(AddRoundKey) is a matrix that contains an ARC(AddRoundConstant) array in
// each row. Each constant is added to the |state| of each round of Poseidon.
// MDS(Maximum Distance Separable) is a matrix that is applied to the |state|
// for each round. It ensures that the sum of the vector's weight before and
// after the MDS is at least |state| + 1.
template <typename PrimeField>
void FindPoseidonArkAndMds(const PoseidonGrainLFSRConfig& config,
                           size_t skip_matrices, math::Matrix<PrimeField>& ark,
                           math::Matrix<PrimeField>& mds) {
  PoseidonGrainLFSR<PrimeField> lfsr(config);
  ark = math::Matrix<PrimeField>(
      config.num_full_rounds + config.num_partial_rounds, config.state_len);
  for (size_t i = 0; i < config.num_full_rounds + config.num_partial_rounds;
       ++i) {
    ark.row(i) = lfsr.GetFieldElementsRejectionSampling(config.state_len);
  }

  for (uint64_t i = 0; i < skip_matrices; ++i) {
    lfsr.GetFieldElementsModP(2 * config.state_len);
  }

  // a qualifying matrix must satisfy the following requirements
  // - there is no duplication among the elements in x or y
  // - there is no i and j such that x[i] + y[j] = p
  // - the resultant MDS passes all the three tests

  math::Vector<PrimeField> xs = lfsr.GetFieldElementsModP(config.state_len);
  math::Vector<PrimeField> ys = lfsr.GetFieldElementsModP(config.state_len);

  mds = math::Matrix<PrimeField>(config.state_len, config.state_len);
  for (Eigen::Index i = 0; i < mds.rows(); ++i) {
    for (Eigen::Index j = 0; j < mds.cols(); ++j) {
      mds(i, j) = unwrap((xs[i] + ys[j]).Inverse());
    }
  }
}

template <typename PrimeField>
struct PoseidonConfig : public PoseidonConfigBase<PrimeField> {
  using F = PrimeField;

  // Maximally Distance Separating (MDS) Matrix.
  math::Matrix<PrimeField> mds;

  PoseidonConfig() = default;
  PoseidonConfig(const PoseidonConfigBase<PrimeField>& base,
                 const math::Matrix<PrimeField>& mds)
      : PoseidonConfigBase<PrimeField>(base), mds(mds) {}
  PoseidonConfig(PoseidonConfigBase<PrimeField>&& base,
                 math::Matrix<PrimeField>&& mds)
      : PoseidonConfigBase<PrimeField>(std::move(base)), mds(std::move(mds)) {}

  static PoseidonConfig CreateDefault(size_t rate, bool optimized_for_weights) {
    absl::Span<const PoseidonConfigEntry> param_set =
        optimized_for_weights ? kPoseidonOptimizedWeightsDefaultParams
                              : kPoseidonOptimizedConstraintsDefaultParams;

    auto it = base::ranges::find_if(param_set.begin(), param_set.end(),
                                    [rate](const PoseidonConfigEntry& param) {
                                      return param.rate == rate;
                                    });
    CHECK_NE(it, param_set.end());
    PoseidonConfig ret = it->template ToPoseidonConfig<PrimeField>();
    FindPoseidonArkAndMds<PrimeField>(
        it->template ToPoseidonGrainLFSRConfig<PrimeField>(), it->skip_matrices,
        ret.ark, ret.mds);
    return ret;
  }

  constexpr static PoseidonConfig CreateCustom(size_t rate, uint64_t alpha,
                                               size_t full_rounds,
                                               size_t partial_rounds,
                                               size_t skip_matrices) {
    PoseidonConfigEntry config_entry(rate, alpha, full_rounds, partial_rounds,
                                     skip_matrices);
    PoseidonConfig ret = config_entry.ToPoseidonConfig<PrimeField>();
    FindPoseidonArkAndMds<PrimeField>(
        config_entry.ToPoseidonGrainLFSRConfig<PrimeField>(), skip_matrices,
        ret.ark, ret.mds);
    return ret;
  }

  bool IsValid() const override {
    return PoseidonConfigBase<PrimeField>::IsValid() &&
           static_cast<size_t>(mds.rows()) == this->rate + this->capacity &&
           static_cast<size_t>(mds.cols()) == this->rate + this->capacity;
  }

  bool operator==(const PoseidonConfig& other) const {
    return PoseidonConfigBase<PrimeField>::operator==(other) &&
           mds == other.mds;
  }
  bool operator!=(const PoseidonConfig& other) const {
    return !operator==(other);
  }
};

template <typename PrimeField>
PoseidonConfig<PrimeField> PoseidonConfigEntry::ToPoseidonConfig() const {
  PoseidonConfig<PrimeField> config;
  config.full_rounds = full_rounds;
  config.partial_rounds = partial_rounds;
  config.alpha = alpha;
  config.rate = rate;
  config.capacity = 1;
  return config;
}

}  // namespace crypto

namespace base {

template <typename PrimeField>
class Copyable<crypto::PoseidonConfig<PrimeField>> {
 public:
  static bool WriteTo(const crypto::PoseidonConfig<PrimeField>& config,
                      Buffer* buffer) {
    return Copyable<crypto::PoseidonConfigBase<PrimeField>>::WriteTo(config,
                                                                     buffer) &&
           buffer->Write(config.mds);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonConfig<PrimeField>* config) {
    crypto::PoseidonConfigBase<PrimeField> base;
    math::Matrix<PrimeField> mds;
    if (!buffer.ReadMany(&base, &mds)) {
      return false;
    }

    *config = {std::move(base), std::move(mds)};
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonConfig<PrimeField>& config) {
    const crypto::PoseidonConfigBase<PrimeField>& base =
        static_cast<const crypto::PoseidonConfigBase<PrimeField>&>(config);
    return base::EstimateSize(base, config.mds);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_
