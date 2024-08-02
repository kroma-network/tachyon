// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_EVALUATOR_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_EVALUATOR_H_

#include <vector>

#include "tachyon/zk/plonk/permutation/permutation_prover.h"
#include "tachyon/zk/plonk/vanishing/circuit_polynomial_builder_forward.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"

namespace tachyon::zk::plonk {

template <typename EvalsOrExtendedEvals>
class PermutationEvaluator {
 public:
  using F = typename EvalsOrExtendedEvals::Field;

  template <typename PS>
  void Evaluate(CircuitPolynomialBuilder<PS>& builder, absl::Span<F> chunk,
                size_t chunk_offset, size_t chunk_size) {
    if (product_cosets_.empty()) return;

    const std::vector<AnyColumnKey>& column_keys =
        builder.proving_key_.verifying_key()
            .constraint_system()
            .permutation()
            .columns();
    std::vector<std::vector<base::Ref<const EvalsOrExtendedEvals>>>
        column_chunks = base::Map(
            base::Chunked(column_keys, builder.chunk_len_),
            [&builder](absl::Span<const AnyColumnKey> column_key_chunk) {
              return builder.table_.GetColumns(column_key_chunk);
            });
    std::vector<absl::Span<const EvalsOrExtendedEvals>> coset_chunks =
        base::Map(
            base::Chunked(cosets_, builder.chunk_len_),
            [](absl::Span<const EvalsOrExtendedEvals> chunk) { return chunk; });

    size_t start = chunk_offset * chunk_size;
    F beta_term = builder.current_extended_omega_ * builder.omega_.Pow(start);
    for (size_t i = 0; i < chunk.size(); ++i) {
      size_t idx = start + i;

      // Enforce only for the first set: l_first(X) * (1 - z₀(X)) = 0
      chunk[i] *= builder.y_;
      chunk[i] +=
          (F::One() - product_cosets_.front()[idx]) * builder.l_first_[idx];

      // Enforce only for the last set: l_last(X) * (zₗ(X)² - zₗ(X)) = 0
      const EvalsOrExtendedEvals& last_coset = product_cosets_.back();
      chunk[i] *= builder.y_;
      chunk[i] +=
          builder.l_last_[idx] * (last_coset[idx].Square() - last_coset[idx]);

      // Except for the first set, enforce:
      // l_first(X) * (zⱼ(X) - zⱼ₋₁(ω⁻¹X)) = 0
      RowIndex r_last =
          builder.last_rotation_.GetIndex(idx, /*scale=*/1, builder.n_);
      for (size_t j = 0; j < product_cosets_.size(); ++j) {
        if (j == 0) continue;
        chunk[i] *= builder.y_;
        chunk[i] += builder.l_first_[idx] *
                    (product_cosets_[j][idx] - product_cosets_[j - 1][r_last]);
      }

      // And for all the sets we enforce: (1 - (l_last(X) + l_blind(X))) *
      // (zⱼ(ωX) * Πⱼ(p(X) + βsⱼ(X) + γ) - zⱼ(X) Πⱼ(p(X) + δʲβX + γ))
      F current_delta = builder.delta_start_ * beta_term;
      RowIndex r_next = Rotation(1).GetIndex(idx, /*scale=*/1, builder.n_);

      for (size_t j = 0; j < product_cosets_.size(); ++j) {
        const std::vector<base::Ref<const EvalsOrExtendedEvals>>& column_chunk =
            column_chunks[j];
        absl::Span<const EvalsOrExtendedEvals> coset_chunk = coset_chunks[j];
        F left = product_cosets_[j][r_next];
        for (size_t k = 0; k < column_chunk.size(); ++k) {
          left *= (*column_chunk[k])[idx] +
                  builder.beta_ * coset_chunk[k][idx] + builder.gamma_;
        }
        F right = product_cosets_[j][idx];
        for (size_t k = 0; k < column_chunk.size(); ++k) {
          right *= (*column_chunk[k])[idx] + current_delta + builder.gamma_;
          current_delta *= builder.delta_;
        }
        chunk[i] *= builder.y_;
        chunk[i] += (left - right) * builder.l_active_row_[idx];
      }
      beta_term *= builder.omega_;
    }
  }

  template <typename PS>
  void UpdateCosets(CircuitPolynomialBuilder<PS>& builder, size_t circuit_idx) {
    using PCS = typename PS::PCS;
    using Poly = typename PCS::Poly;
    using Evals = typename PCS::Evals;

    constexpr halo2::Vendor kVendor = PS::kVendor;

    size_t num_permutations =
        builder.permutation_provers_[circuit_idx].grand_product_polys().size();
    if (num_permutations == 0) return;

    const PermutationProver<Poly, Evals>& prover =
        builder.permutation_provers_[circuit_idx];
    product_cosets_.resize(num_permutations);
    for (size_t i = 0; i < num_permutations; ++i) {
      if constexpr (kVendor == halo2::Vendor::kPSE) {
        product_cosets_[i] = CoeffToExtended(
            prover.grand_product_polys()[i].poly(), builder.extended_domain_);
      } else {
        product_cosets_[i] =
            builder.coset_domain_->FFT(prover.grand_product_polys()[i].poly());
      }
    }

    if constexpr (kVendor == halo2::Vendor::kScroll) {
      const std::vector<Poly>& polys =
          builder.proving_key_.permutation_proving_key().polys();
      cosets_.resize(polys.size());
      for (size_t i = 0; i < polys.size(); ++i) {
        cosets_[i] = builder.coset_domain_->FFT(polys[i]);
      }
    }
  }

 private:
  std::vector<EvalsOrExtendedEvals> product_cosets_;
  std::vector<EvalsOrExtendedEvals> cosets_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_EVALUATOR_H_
