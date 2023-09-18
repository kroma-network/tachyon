#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_

#include <vector>

#include "tachyon/crypto/hashes/sponge/poseidon/grain_lfsr.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"
#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field_forward.h"

namespace tachyon::crypto {

// An entry in the default Poseidon parameters
class PoseidonDefaultConfigEntry {
 public:
  // The rate (in terms of number of field elements).
  size_t rate;

  // Exponent used in S-boxes.
  size_t alpha;

  // Number of rounds in a full-round operation.
  size_t full_rounds;

  // Number of rounds in a partial-round operation.
  size_t partial_rounds;

  // Number of matrices to skip when generating parameters using the Grain LFSR.
  // The matrices being skipped are those that do not satisfy all the desired
  // properties. See:
  // https://extgit.iaik.tugraz.at/krypto/hadeshash/-/blob/master/code/generate_parameters_grain.sage
  size_t skip_matrices;

  PoseidonDefaultConfigEntry(size_t rate, size_t alpha, size_t full_rounds,
                             size_t partial_rounds, size_t skip_matrices)
      : rate(rate),
        alpha(alpha),
        full_rounds(full_rounds),
        partial_rounds(partial_rounds),
        skip_matrices(skip_matrices){};
};

class PoseidonDefaultConfig {
 public:
  // An array of the parameters optimized for constraints
  // (rate, alpha, full_rounds, partial_rounds, skip_matrices)
  // for rate = 2, 3, 4, 5, 6, 7, 8
  // Here, `skip_matrices` denotes how many matrices to skip before finding one
  // that satisfy all the requirements.
  const std::vector<PoseidonDefaultConfigEntry> params_opt_for_constraints = {
      PoseidonDefaultConfigEntry(2, 17, 8, 31, 0),
      PoseidonDefaultConfigEntry(3, 5, 8, 56, 0),
      PoseidonDefaultConfigEntry(4, 5, 8, 56, 0),
      PoseidonDefaultConfigEntry(5, 5, 8, 57, 0),
      PoseidonDefaultConfigEntry(6, 5, 8, 57, 0),
      PoseidonDefaultConfigEntry(7, 5, 8, 57, 0),
      PoseidonDefaultConfigEntry(8, 5, 8, 57, 0)};
  // An array of the parameters optimized for weights
  // (rate, alpha, full_rounds, partial_rounds, skip_matrices)
  // for rate = 2, 3, 4, 5, 6, 7, 8
  const std::vector<PoseidonDefaultConfigEntry> params_opt_for_weights = {
      PoseidonDefaultConfigEntry(2, 257, 8, 13, 0),
      PoseidonDefaultConfigEntry(3, 257, 8, 13, 0),
      PoseidonDefaultConfigEntry(4, 257, 8, 13, 0),
      PoseidonDefaultConfigEntry(5, 257, 8, 13, 0),
      PoseidonDefaultConfigEntry(6, 257, 8, 13, 0),
      PoseidonDefaultConfigEntry(7, 257, 8, 13, 0),
      PoseidonDefaultConfigEntry(8, 257, 8, 13, 0)};
};

template <typename PrimeFieldTy>
std::pair<std::vector<std::vector<PrimeFieldTy>>,
          std::vector<std::vector<PrimeFieldTy>>>
FindPoseidonArkAndMds(uint64_t prime_bits, size_t rate, size_t full_rounds,
                      size_t partial_rounds, size_t skip_matrices) {
  PoseidonGrainLFSR<PrimeFieldTy> lfsr(false, prime_bits, uint64_t(rate + 1),
                                       full_rounds, partial_rounds);

  std::vector<std::vector<PrimeFieldTy>> ark;
  for (size_t i = 0; i < full_rounds + partial_rounds; ++i) {
    ark.emplace_back(lfsr.GetFieldElementsRejectionSampling(rate + 1));
  }

  std::vector<std::vector<PrimeFieldTy>> mds(
      rate + 1, std::vector<PrimeFieldTy>(rate + 1, PrimeFieldTy::Zero()));
  for (uint64_t i = 0; i < skip_matrices; ++i) {
    lfsr.GetFieldElementsModP(2 * (rate + 1));
  }

  // a qualifying matrix must satisfy the following requirements
  // - there is no duplication among the elements in x or y
  // - there is no i and j such that x[i] + y[j] = p
  // - the resultant MDS passes all the three tests

  auto xs = lfsr.GetFieldElementsModP(rate + 1);
  auto ys = lfsr.GetFieldElementsModP(rate + 1);

  for (size_t i = 0; i < (rate + 1); ++i) {
    for (size_t j = 0; j < (rate + 1); ++j) {
      mds[i][j] = (xs[i] + ys[j]).Inverse();
    }
  }

  return std::make_pair(ark, mds);
}

template <typename Config>
PoseidonConfig<math::PrimeField<Config>> GetDefaultPoseidonParameters(
    size_t rate, bool optimized_for_weights) {
  using PrimeFieldTy = math::PrimeField<Config>;

  PoseidonDefaultConfig default_config;
  std::vector<PoseidonDefaultConfigEntry> param_set;
  if (optimized_for_weights) {
    param_set = default_config.params_opt_for_weights;
  } else {
    param_set = default_config.params_opt_for_constraints;
  }

  for (auto& param : param_set) {
    if (param.rate == rate) {
      std::pair<std::vector<std::vector<PrimeFieldTy>>,
                std::vector<std::vector<PrimeFieldTy>>>
          ark_and_mds = FindPoseidonArkAndMds<PrimeFieldTy>(
              PrimeFieldTy::kModulusBits, param.rate, param.full_rounds,
              param.partial_rounds, param.skip_matrices);

      PoseidonConfig<PrimeFieldTy> poseidon_config(
          param.full_rounds, param.partial_rounds, uint64_t(param.alpha),
          ark_and_mds.first, ark_and_mds.second, param.rate, size_t(1));
      return poseidon_config;
    }
  }
  throw std::runtime_error("No matching Poseidon parameters found");
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_
