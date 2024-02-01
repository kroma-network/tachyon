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

namespace tachyon {

namespace zk::plonk {

template <typename Poly, typename Evals, typename C>
class ProvingKey : public Key {
 public:
  using F = typename Poly::Field;

  ProvingKey() = default;

  const VerifyingKey<F, C>& verifying_key() const { return verifying_key_; }
  const Poly& l_first() const { return l_first_; }
  const Poly& l_last() const { return l_last_; }
  const Poly& l_active_row() const { return l_active_row_; }
  const std::vector<Evals>& fixed_columns() const { return fixed_columns_; }
  const std::vector<Poly>& fixed_polys() const { return fixed_polys_; }
  const PermutationProvingKey<Poly, Evals>& permutation_proving_key() const {
    return permutation_proving_key_;
  }

  // Return true if it is able to load from an instance of |circuit|.
  template <typename PCS, typename Circuit>
  [[nodiscard]] bool Load(ProverBase<PCS>* prover, const Circuit& circuit) {
    using RationalEvals = typename PCS::RationalEvals;
    KeyPreLoadResult<Evals, RationalEvals> pre_load_result;
    if (!this->PreLoad(prover, circuit, &pre_load_result)) return false;
    VerifyingKeyLoadResult<Evals> vk_result;
    if (!verifying_key_.DoLoad(prover, std::move(pre_load_result), &vk_result))
      return false;
    return DoLoad(prover, std::move(pre_load_result), &vk_result);
  }

  // Return true if it is able to load from an instance of |circuit| and a
  // |verifying_key|.
  template <typename PCS, typename Circuit>
  [[nodiscard]] bool LoadWithVerifyingKey(ProverBase<PCS>* prover,
                                          const Circuit& circuit,
                                          VerifyingKey<F, C>&& verifying_key) {
    using RationalEvals = typename PCS::RationalEvals;
    KeyPreLoadResult<Evals, RationalEvals> pre_load_result;
    if (!this->PreLoad(prover, circuit, &pre_load_result)) return false;
    verifying_key_ = std::move(verifying_key);
    return DoLoad(prover, std::move(pre_load_result), nullptr);
  }

 private:
  friend class c::zk::ProvingKeyImplBase<Poly, Evals, C>;

  template <typename PCS, typename RationalEvals>
  bool DoLoad(ProverBase<PCS>* prover,
              KeyPreLoadResult<Evals, RationalEvals>&& pre_load_result,
              VerifyingKeyLoadResult<Evals>* vk_load_result) {
    using Domain = typename PCS::Domain;

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
      permutations = pre_load_result.assembly.permutation()
                         .template GeneratePermutations<Evals>(domain);
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
    // | 4 | 1         |
    // | 5 | 0         |
    // | 6 | 0         |
    // | 7 | 0         |
    RowIndex usable_rows = prover->GetUsableRows();
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

    vanishing_argument_ =
        VanishingArgument<F>::Create(verifying_key_.constraint_system());
    return true;
  }

  VerifyingKey<F, C> verifying_key_;
  Poly l_first_;
  Poly l_last_;
  Poly l_active_row_;
  std::vector<Evals> fixed_columns_;
  std::vector<Poly> fixed_polys_;
  PermutationProvingKey<Poly, Evals> permutation_proving_key_;
  VanishingArgument<F> vanishing_argument_;
};

}  // namespace zk::plonk
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
