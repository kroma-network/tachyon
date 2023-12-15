// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_

#include <utility>
#include <vector>

#include "tachyon/base/parallelize.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"
#include "tachyon/zk/plonk/vanishing/vanishing_argument.h"

namespace tachyon::zk {

template <typename PCSTy>
class ProvingKey : public Key<PCSTy> {
 public:
  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;
  using Evals = typename PCSTy::Evals;
  using PreLoadResult = typename Key<PCSTy>::PreLoadResult;
  using VerifyingKeyLoadResult = typename VerifyingKey<PCSTy>::LoadResult;

  ProvingKey() = default;

  const VerifyingKey<PCSTy>& verifying_key() const { return verifying_key_; }
  const Poly& l_first() const { return l_first_; }
  const Poly& l_last() const { return l_last_; }
  const Poly& l_active_row() const { return l_active_row_; }
  const std::vector<Evals>& fixed_columns() const { return fixed_columns_; }
  const std::vector<Poly>& fixed_polys() const { return fixed_polys_; }
  const PermutationProvingKey<Poly, Evals>& permutation_proving_key() const {
    return permutation_proving_key_;
  }

  // Return true if it is able to load from an instance of |circuit|.
  template <typename CircuitTy>
  [[nodiscard]] bool Load(ProverBase<PCSTy>* prover, const CircuitTy& circuit) {
    PreLoadResult pre_load_result;
    if (!this->PreLoad(prover, circuit, &pre_load_result)) return false;
    VerifyingKeyLoadResult vk_result;
    if (!verifying_key_.DoLoad(prover, std::move(pre_load_result), &vk_result))
      return false;
    return DoLoad(prover, std::move(pre_load_result), &vk_result);
  }

  // Return true if it is able to load from an instance of |circuit| and a
  // |verifying_key|.
  template <typename CircuitTy>
  [[nodiscard]] bool LoadWithVerifyingKey(ProverBase<PCSTy>* prover,
                                          const CircuitTy& circuit,
                                          VerifyingKey<PCSTy>&& verifying_key) {
    PreLoadResult pre_load_result;
    if (!this->PreLoad(prover, circuit, &pre_load_result)) return false;
    verifying_key_ = std::move(verifying_key);
    return DoLoad(prover, std::move(pre_load_result), nullptr);
  }

 private:
  bool DoLoad(ProverBase<PCSTy>* prover, PreLoadResult&& pre_load_result,
              VerifyingKeyLoadResult* vk_load_result) {
    using Domain = typename PCSTy::Domain;

    // NOTE(chokobole): |ComputeBlindingFactors()| is a second call. The first
    // was called inside |PreLoad()|. But I think this is cheap to compute.
    prover->blinder().set_blinding_factors(
        verifying_key_.constraint_system().ComputeBlindingFactors());

    const Domain* domain = prover->domain();
    fixed_columns_ = std::move(pre_load_result.fixed_columns);
    fixed_polys_ = base::Map(fixed_columns_, [domain](const Evals& evals) {
      return domain->IFFT(evals);
    });

    std::vector<Evals> permutations;
    if (vk_load_result) {
      permutations = std::move(vk_load_result->permutations);
    } else {
      permutations =
          pre_load_result.assembly.permutation().GeneratePermutations(domain);
    }

    permutation_proving_key_ =
        pre_load_result.assembly.permutation().BuildProvingKey(prover,
                                                               permutations);

    // Compute l_first(X)
    // if |blinding_factors| = 3 and |pcs.N()| = 8,
    //
    // | X | l_first(X) |
    // |---|------------|
    // | 0 | 1          |
    // | 1 | 0          |
    // | 2 | 0          |
    // | 3 | 0          |
    // | 4 | 0          |
    // | 5 | 0          |
    // | 6 | 0          |
    // | 7 | 0          |
    Evals evals = domain->template Empty<Evals>();
    *evals[0] = F::One();
    l_first_ = domain->IFFT(evals);
    *evals[0] = F::Zero();

    // Compute l_last(X) which evaluates to 1 on the first inactive row (just
    // before the blinding factors) and 0 otherwise over the domain.
    //
    // | X | l_last(X) |
    // |---|-----------|
    // | 0 | 0         |
    // | 1 | 0         |
    // | 2 | 0         |
    // | 3 | 0         |
    // | 4 | 0         |
    // | 5 | 0         |
    // | 6 | 0         |
    // | 7 | 1         |
    size_t usable_rows = prover->GetUsableRows();
    *evals[usable_rows] = F::One();
    l_last_ = domain->IFFT(evals);
    *evals[usable_rows] = F::Zero();

    // Compute l_active_row(X).
    //
    // | X | l_active_row(X) |
    // |---|-----------------|
    // | 0 | 1               |
    // | 1 | 1               |
    // | 2 | 1               |
    // | 3 | 1               |
    // | 4 | 0               |
    // | 5 | 0               |
    // | 6 | 0               |
    // | 7 | 0               |
    F one = F::One();
    base::Parallelize(
        evals.evaluations(),
        [&one, usable_rows](absl::Span<F> chunk, size_t chunk_index,
                            size_t chunk_size_in) {
          size_t i = chunk_index * chunk_size_in;
          for (F& value : chunk) {
            if (i++ < usable_rows) {
              value = one;
            }
          }
        });
    l_active_row_ = domain->IFFT(evals);

    // TODO(chokobole): Set |vanishing_argument_|.
    // See
    // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/keygen.rs#L395.
    return true;
  }

  VerifyingKey<PCSTy> verifying_key_;
  Poly l_first_;
  Poly l_last_;
  Poly l_active_row_;
  std::vector<Evals> fixed_columns_;
  std::vector<Poly> fixed_polys_;
  PermutationProvingKey<Poly, Evals> permutation_proving_key_;
  VanishingArgument<PCSTy> vanishing_argument_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
