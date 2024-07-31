// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_

#include <type_traits>
#include <utility>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/halo2/vendor.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"
#include "tachyon/zk/plonk/vanishing/vanishing_argument.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"

namespace tachyon {

namespace zk::plonk {

template <halo2::Vendor Vendor, typename LS>
class ProvingKey : public Key {
 public:
  using F = typename LS::Field;
  using Poly = typename LS::Poly;
  using Evals = typename LS::Evals;
  using ExtendedEvals = typename LS::ExtendedEvals;
  using C = typename LS::Commitment;

  using PolyOrExtendedEvals =
      std::conditional_t<Vendor == halo2::Vendor::kPSE, ExtendedEvals, Poly>;

  ProvingKey() = default;

  const VerifyingKey<F, C>& verifying_key() const { return verifying_key_; }
  const PolyOrExtendedEvals& l_first() const { return l_first_; }
  const PolyOrExtendedEvals& l_last() const { return l_last_; }
  const PolyOrExtendedEvals& l_active_row() const { return l_active_row_; }
  const std::vector<Evals>& fixed_columns() const { return fixed_columns_; }
  std::vector<Evals>& fixed_columns() { return fixed_columns_; }
  const std::vector<Poly>& fixed_polys() const { return fixed_polys_; }
  const std::vector<PolyOrExtendedEvals>& fixed_cosets() const {
    return fixed_cosets_;
  }
  const PermutationProvingKey<Poly, Evals, ExtendedEvals>&
  permutation_proving_key() const {
    return permutation_proving_key_;
  }

  // Return true if it is able to load from an instance of |circuit|.
  template <typename PCS, typename Circuit>
  [[nodiscard]] bool Load(ProverBase<PCS>* prover, const Circuit& circuit) {
    using RationalEvals = typename PCS::RationalEvals;
    KeyPreLoadResult<Evals, RationalEvals> pre_load_result(LS::type);
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
    KeyPreLoadResult<Evals, RationalEvals> pre_load_result(LS::type);
    if (!this->PreLoad(prover, circuit, &pre_load_result)) return false;
    verifying_key_ = std::move(verifying_key);
    return DoLoad(prover, std::move(pre_load_result), nullptr);
  }

 private:
  friend class c::zk::plonk::ProvingKeyImpl<Vendor, LS>;

  template <typename PCS, typename RationalEvals>
  bool DoLoad(ProverBase<PCS>* prover,
              KeyPreLoadResult<Evals, RationalEvals>&& pre_load_result,
              VerifyingKeyLoadResult<Evals>* vk_load_result) {
    using Domain = typename PCS::Domain;

    // NOTE(chokobole): |ComputeBlindingFactors()| is a second call. The first
    // was called inside |PreLoad()|. But it's okay since this is cheap to
    // compute.
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
        pre_load_result.assembly.permutation().template BuildProvingKey<Vendor>(
            prover, std::move(permutations));

    // Compute l_first(X).
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
    Evals evals = domain->template Zero<Evals>();
    // NOTE(chokobole): It's safe to access since we created |domain->size()|
    // |evals|.
    evals.at(0) = F::One();
    Poly l_first = domain->IFFT(evals);
    evals.at(0) = F::Zero();

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
    // NOTE(chokobole): It's safe to access since we created |domain->size()|
    // |evals|, which is greater than |usable_rows|.
    evals.at(usable_rows) = F::One();
    Poly l_last = domain->IFFT(evals);
    evals.at(usable_rows) = F::Zero();

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
    OPENMP_PARALLEL_FOR(size_t i = 0; i < usable_rows; ++i) {
      // NOTE(chokobole): It's safe to access since we created |domain->size()|
      // |evals|, which is greater than |usable_rows|.
      evals.at(i) = F::One();
    }
    Poly l_active_row = domain->IFFT(std::move(evals));

    if constexpr (Vendor == halo2::Vendor::kPSE) {
      using ExtendedDomain = typename PCS::ExtendedDomain;

      const ExtendedDomain* extended_domain = prover->extended_domain();
      l_first_ = CoeffToExtended(std::move(l_first), extended_domain);
      l_last_ = CoeffToExtended(std::move(l_last), extended_domain);
      l_active_row_ = CoeffToExtended(std::move(l_active_row), extended_domain);

      fixed_cosets_ =
          base::Map(fixed_polys_, [domain, extended_domain](const Poly& poly) {
            return CoeffToExtended(poly, extended_domain);
          });
    } else {
      l_first_ = std::move(l_first);
      l_last_ = std::move(l_last);
      l_active_row_ = std::move(l_active_row);
    }

    vanishing_argument_ = VanishingArgument<Vendor, LS>::Create(
        verifying_key_.constraint_system());
    return true;
  }

  VerifyingKey<F, C> verifying_key_;
  PolyOrExtendedEvals l_first_;
  PolyOrExtendedEvals l_last_;
  PolyOrExtendedEvals l_active_row_;
  std::vector<Evals> fixed_columns_;
  std::vector<Poly> fixed_polys_;
  // NOTE(chokobole): Only PSE Halo2 has the member |fixed_cosets_|.
  // See below:
  // https://github.com/privacy-scaling-explorations/halo2/blob/bc857a7/halo2_backend/src/plonk.rs#L260-L270
  // https://github.com/scroll-tech/halo2/blob/1070391/halo2_proofs/src/plonk.rs#L263-L272
  std::vector<ExtendedEvals> fixed_cosets_;
  PermutationProvingKey<Poly, Evals, ExtendedEvals> permutation_proving_key_;
  VanishingArgument<Vendor, LS> vanishing_argument_;
};

}  // namespace zk::plonk
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
