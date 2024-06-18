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
template <typename F>
void FindPoseidonArkAndMds(const PoseidonGrainLFSRConfig& config,
                           size_t skip_matrices, math::Matrix<F>& ark,
                           math::Matrix<F>& mds) {
  PoseidonGrainLFSR<F> lfsr(config);
  ark = math::Matrix<F>(config.num_full_rounds + config.num_partial_rounds,
                        config.state_len);
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

  math::Vector<F> xs = lfsr.GetFieldElementsModP(config.state_len);
  math::Vector<F> ys = lfsr.GetFieldElementsModP(config.state_len);

  mds = math::Matrix<F>(config.state_len, config.state_len);
  for (Eigen::Index i = 0; i < mds.rows(); ++i) {
    for (Eigen::Index j = 0; j < mds.cols(); ++j) {
      mds(i, j) = unwrap((xs[i] + ys[j]).Inverse());
    }
  }
}

template <typename F>
struct PoseidonConfig : public PoseidonConfigBase<F> {
  // Maximally Distance Separating (MDS) Matrix.
  math::Matrix<F> mds;

  PoseidonConfig() = default;
  PoseidonConfig(const PoseidonConfigBase<F>& base, const math::Matrix<F>& mds)
      : PoseidonConfigBase<F>(base), mds(mds) {}
  PoseidonConfig(PoseidonConfigBase<F>&& base, math::Matrix<F>&& mds)
      : PoseidonConfigBase<F>(std::move(base)), mds(std::move(mds)) {}

  static PoseidonConfig CreateDefault(size_t rate, bool optimized_for_weights) {
    absl::Span<const PoseidonConfigEntry> param_set =
        optimized_for_weights ? kPoseidonOptimizedWeightsDefaultParams
                              : kPoseidonOptimizedConstraintsDefaultParams;

    auto it = base::ranges::find_if(param_set.begin(), param_set.end(),
                                    [rate](const PoseidonConfigEntry& param) {
                                      return param.rate == rate;
                                    });
    CHECK_NE(it, param_set.end());
    PoseidonConfig ret = it->template ToPoseidonConfig<F>();
    FindPoseidonArkAndMds<F>(it->template ToPoseidonGrainLFSRConfig<F>(),
                             it->skip_matrices, ret.ark, ret.mds);
    return ret;
  }

  constexpr static PoseidonConfig CreateCustom(size_t rate, uint64_t alpha,
                                               size_t full_rounds,
                                               size_t partial_rounds,
                                               size_t skip_matrices) {
    PoseidonConfigEntry config_entry(rate, alpha, full_rounds, partial_rounds,
                                     skip_matrices);
    PoseidonConfig ret = config_entry.ToPoseidonConfig<F>();
    FindPoseidonArkAndMds<F>(config_entry.ToPoseidonGrainLFSRConfig<F>(),
                             skip_matrices, ret.ark, ret.mds);
    return ret;
  }

  bool IsValid() const override {
    return PoseidonConfigBase<F>::IsValid() &&
           static_cast<size_t>(mds.rows()) == this->rate + this->capacity &&
           static_cast<size_t>(mds.cols()) == this->rate + this->capacity;
  }

  bool operator==(const PoseidonConfig& other) const {
    return PoseidonConfigBase<F>::operator==(other) && mds == other.mds;
  }
  bool operator!=(const PoseidonConfig& other) const {
    return !operator==(other);
  }
};

template <typename F>
PoseidonConfig<F> PoseidonConfigEntry::ToPoseidonConfig() const {
  PoseidonConfig<F> config;
  config.full_rounds = full_rounds;
  config.partial_rounds = partial_rounds;
  config.alpha = alpha;
  config.rate = rate;
  config.capacity = 1;
  return config;
}

}  // namespace crypto

namespace base {

template <typename F>
class Copyable<crypto::PoseidonConfig<F>> {
 public:
  static bool WriteTo(const crypto::PoseidonConfig<F>& config, Buffer* buffer) {
    return Copyable<crypto::PoseidonConfigBase<F>>::WriteTo(config, buffer) &&
           buffer->Write(config.mds);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonConfig<F>* config) {
    crypto::PoseidonConfigBase<F> base;
    math::Matrix<F> mds;
    if (!buffer.ReadMany(&base, &mds)) {
      return false;
    }

    *config = {std::move(base), std::move(mds)};
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonConfig<F>& config) {
    const crypto::PoseidonConfigBase<F>& base =
        static_cast<const crypto::PoseidonConfigBase<F>&>(config);
    return base::EstimateSize(base, config.mds);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_
