// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_

#include "absl/types/span.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/crypto/hashes/sponge/poseidon/grain_lfsr.h"

namespace tachyon::crypto {

template <typename PrimeFieldTy>
struct PoseidonConfig;

// An entry in the Poseidon config
struct TACHYON_EXPORT PoseidonConfigEntry {
  // The rate (in terms of number of field elements).
  size_t rate;

  // Exponent used in S-boxes.
  uint64_t alpha;

  // Number of rounds in a full-round operation.
  size_t full_rounds;

  // Number of rounds in a partial-round operation.
  size_t partial_rounds;

  // Number of matrices to skip when generating config using the Grain LFSR.
  // The matrices being skipped are those that do not satisfy all the desired
  // properties. See:
  // https://extgit.iaik.tugraz.at/krypto/hadeshash/-/blob/master/code/generate_parameters_grain.sage
  size_t skip_matrices;

  constexpr PoseidonConfigEntry() : PoseidonConfigEntry(0, 0, 0, 0, 0) {}
  constexpr PoseidonConfigEntry(size_t rate, uint64_t alpha, size_t full_rounds,
                                size_t partial_rounds, size_t skip_matrices)
      : rate(rate),
        alpha(alpha),
        full_rounds(full_rounds),
        partial_rounds(partial_rounds),
        skip_matrices(skip_matrices) {}

  template <typename PrimeFieldTy>
  PoseidonGrainLFSRConfig ToPoseidonGrainLFSRConfig() const {
    PoseidonGrainLFSRConfig config;
    config.prime_num_bits = PrimeFieldTy::kModulusBits;
    config.state_len = rate + 1;
    config.num_full_rounds = full_rounds;
    config.num_partial_rounds = partial_rounds;
    return config;
  }

  template <typename PrimeFieldTy>
  PoseidonConfig<PrimeFieldTy> ToPoseidonConfig() const;
};

// An array of the default config optimized for constraints
// (rate, alpha, full_rounds, partial_rounds, skip_matrices)
// for rate = 2, 3, 4, 5, 6, 7, 8
// Here, |skip_matrices| denotes how many matrices to skip before finding one
// that satisfy all the requirements.
constexpr const PoseidonConfigEntry kOptimizedConstraintsDefaultParams[] = {
    PoseidonConfigEntry(2, 17, 8, 31, 0), PoseidonConfigEntry(3, 5, 8, 56, 0),
    PoseidonConfigEntry(4, 5, 8, 56, 0),  PoseidonConfigEntry(5, 5, 8, 57, 0),
    PoseidonConfigEntry(6, 5, 8, 57, 0),  PoseidonConfigEntry(7, 5, 8, 57, 0),
    PoseidonConfigEntry(8, 5, 8, 57, 0),
};

// An array of the default config optimized for weights
// (rate, alpha, full_rounds, partial_rounds, skip_matrices)
// for rate = 2, 3, 4, 5, 6, 7, 8
constexpr const PoseidonConfigEntry kOptimizedWeightsDefaultParams[] = {
    PoseidonConfigEntry(2, 257, 8, 13, 0),
    PoseidonConfigEntry(3, 257, 8, 13, 0),
    PoseidonConfigEntry(4, 257, 8, 13, 0),
    PoseidonConfigEntry(5, 257, 8, 13, 0),
    PoseidonConfigEntry(6, 257, 8, 13, 0),
    PoseidonConfigEntry(7, 257, 8, 13, 0),
    PoseidonConfigEntry(8, 257, 8, 13, 0),
};

// ARK(AddRoundKey) is a matrix that contains an ARC(AddRoundConstant) array in
// each row. Each constant is added to the |state| of each round of Poseidon.
// MDS(Maximum Distance Separable) is a matrix that is applied to the |state|
// for each round. It ensures that the sum of the vector's weight before and
// after the MDS is at least |state| + 1.
template <typename PrimeFieldTy>
void FindPoseidonArkAndMds(const PoseidonGrainLFSRConfig& config,
                           size_t skip_matrices,
                           math::Matrix<PrimeFieldTy>* ark,
                           math::Matrix<PrimeFieldTy>* mds) {
  PoseidonGrainLFSR<PrimeFieldTy> lfsr(config);
  *ark = math::Matrix<PrimeFieldTy>(
      config.num_full_rounds + config.num_partial_rounds, config.state_len);
  for (size_t i = 0; i < config.num_full_rounds + config.num_partial_rounds;
       ++i) {
    ark->row(i) = lfsr.GetFieldElementsRejectionSampling(config.state_len);
  }

  for (uint64_t i = 0; i < skip_matrices; ++i) {
    lfsr.GetFieldElementsModP(2 * config.state_len);
  }

  // a qualifying matrix must satisfy the following requirements
  // - there is no duplication among the elements in x or y
  // - there is no i and j such that x[i] + y[j] = p
  // - the resultant MDS passes all the three tests

  math::Vector<PrimeFieldTy> xs = lfsr.GetFieldElementsModP(config.state_len);
  math::Vector<PrimeFieldTy> ys = lfsr.GetFieldElementsModP(config.state_len);

  *mds = math::Matrix<PrimeFieldTy>(config.state_len, config.state_len);
  for (Eigen::Index i = 0; i < mds->rows(); ++i) {
    for (Eigen::Index j = 0; j < mds->cols(); ++j) {
      (*mds)(i, j) = (xs[i] + ys[j]).Inverse();
    }
  }
}

template <typename PrimeFieldTy>
struct PoseidonConfig {
  using F = PrimeFieldTy;

  // Number of rounds in a full-round operation.
  size_t full_rounds = 0;

  // Number of rounds in a partial-round operation.
  size_t partial_rounds = 0;

  // Exponent used in S-boxes.
  uint64_t alpha = 0;

  // Additive Round Keys added before each MDS matrix application to make it an
  // affine shift. They are indexed by |ark[round_num][state_element_index]|.
  math::Matrix<PrimeFieldTy> ark;

  // Maximally Distance Separating (MDS) Matrix.
  math::Matrix<PrimeFieldTy> mds;

  // The rate (in terms of number of field elements).
  // See https://iacr.org/archive/eurocrypt2008/49650180/49650180.pdf
  size_t rate = 0;

  // The capacity (in terms of number of field elements).
  size_t capacity = 0;

  static PoseidonConfig CreateDefault(size_t rate, bool optimized_for_weights) {
    absl::Span<const PoseidonConfigEntry> param_set =
        optimized_for_weights
            ? absl::MakeConstSpan(kOptimizedWeightsDefaultParams)
            : absl::MakeConstSpan(kOptimizedConstraintsDefaultParams);

    auto it = base::ranges::find_if(param_set.begin(), param_set.end(),
                                    [rate](const PoseidonConfigEntry& param) {
                                      return param.rate == rate;
                                    });
    CHECK_NE(it, param_set.end());
    PoseidonConfig ret = it->template ToPoseidonConfig<PrimeFieldTy>();
    FindPoseidonArkAndMds<PrimeFieldTy>(
        it->template ToPoseidonGrainLFSRConfig<PrimeFieldTy>(),
        it->skip_matrices, &ret.ark, &ret.mds);
    return ret;
  }

  bool IsValid() const {
    return static_cast<size_t>(ark.rows()) == full_rounds + partial_rounds &&
           static_cast<size_t>(ark.cols()) == rate + capacity &&
           static_cast<size_t>(mds.rows()) == rate + capacity &&
           static_cast<size_t>(mds.cols()) == rate + capacity;
  }
};

template <typename PrimeFieldTy>
PoseidonConfig<PrimeFieldTy> PoseidonConfigEntry::ToPoseidonConfig() const {
  PoseidonConfig<PrimeFieldTy> config;
  config.full_rounds = full_rounds;
  config.partial_rounds = partial_rounds;
  config.alpha = alpha;
  config.rate = rate;
  config.capacity = 1;
  return config;
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_
