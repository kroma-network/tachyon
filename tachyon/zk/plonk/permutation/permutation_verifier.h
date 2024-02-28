// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFIER_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFIER_H_

#include <vector>

#include "tachyon/base/containers/adapters.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/plonk/base/l_values.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/permutation/permutation_opening_point_set.h"
#include "tachyon/zk/plonk/permutation/permutation_utils.h"
#include "tachyon/zk/plonk/permutation/permutation_verifier_data.h"

namespace tachyon::zk::plonk {

template <typename F, typename C>
class PermutationVerifier {
 public:
  explicit PermutationVerifier(const PermutationVerifierData<F, C>& data)
      : data_(data) {}

  void Evaluate(const ConstraintSystem<F>& constraint_system, const F& x,
                const LValues<F>& l_values, std::vector<F>& evals) const {
    if (data_.grand_product_evals.empty()) return;

    // l_first(x) * (1 - Zₚ,₀(x)) = 0
    const F& z_first = data_.grand_product_evals.front();
    evals.push_back(l_values.first * (F::One() - z_first));
    // l_last(x) * (Zₚ,ₗₐₛₜ(x)² - Zₚ,ₗₐₛₜ(x)) = 0
    const F& z_last = data_.grand_product_evals.back();
    evals.push_back(l_values.last * (z_last.Square() - z_last));
    // l_first(x) * (Zₚ,ᵢ(x) - Zₚ,ᵢ₋₁(ω^(last) * x)) = 0
    for (size_t i = 1; i < data_.grand_product_evals.size(); ++i) {
      const F& z_cur = data_.grand_product_evals[i];
      const F& z_prev_last = *data_.grand_product_last_evals[i - 1];
      evals.push_back(l_values.first * (z_cur - z_prev_last));
    }
    // (1 - (l_last(x) + l_blind(x))) * (
    //   Zₚ,ᵢ(ω * x) Π (vᵢ(x) + β * sᵢ(x) + γ)
    // - Zₚ,ᵢ(x) Π (vᵢ(x) + δⁱ *β * x + γ)
    // ) = 0
    const PermutationArgument& argument = constraint_system.permutation();
    size_t chunk_idx = 0;
    const F& delta = GetDelta<F>();
    F active_rows = F::One() - (l_values.last + l_values.blind);
    size_t chunk_len = constraint_system.ComputePermutationChunkLen();
    for (absl::Span<const AnyColumnKey> columns :
         base::Chunked(argument.columns(), chunk_len)) {
      absl::Span<const F> substitution_evals_chunk =
          data_.substitution_evals.subspan(chunk_idx * chunk_len,
                                           columns.size());
      F left = data_.grand_product_next_evals[chunk_idx];
      for (size_t i = 0; i < columns.size(); ++i) {
        const AnyColumnKey& column = columns[i];
        const F& eval = GetEval(constraint_system, column);
        left *= eval + data_.beta * substitution_evals_chunk[i] + data_.gamma;
      }

      F right = data_.grand_product_evals[chunk_idx];
      F current_delta = data_.beta * x * delta.Pow(chunk_idx * chunk_len);
      for (size_t i = 0; i < columns.size(); ++i) {
        const AnyColumnKey& column = columns[i];
        const F& eval = GetEval(constraint_system, column);
        right *= eval + current_delta + data_.gamma;
        current_delta *= delta;
      }
      evals.push_back(active_rows * (left - right));
      ++chunk_idx;
    }
  }

  template <typename Poly>
  void Open(const PermutationOpeningPointSet<F>& point_set,
            std::vector<crypto::PolynomialOpening<Poly, C>>& openings) const {
    if (data_.grand_product_evals.empty()) return;

#define OPENING(commitment, point, eval) \
  base::Ref<const C>(&data_.commitment), point_set.point, data_.eval

    for (size_t i = 0; i < data_.grand_product_commitments.size(); ++i) {
      openings.emplace_back(
          OPENING(grand_product_commitments[i], x, grand_product_evals[i]));
      openings.emplace_back(OPENING(grand_product_commitments[i], x_next,
                                    grand_product_next_evals[i]));
    }
    if (data_.grand_product_commitments.size() > 1) {
      for (size_t i = data_.grand_product_commitments.size() - 2; i != SIZE_MAX;
           --i) {
        openings.emplace_back(OPENING(grand_product_commitments[i], x_last,
                                      grand_product_last_evals[i].value()));
      }
    }

#undef OPENING
  }

  template <typename Poly>
  void OpenPermutationProvingKey(
      const F& x,
      std::vector<crypto::PolynomialOpening<Poly, C>>& openings) const {
#define OPENING(commitment, point, eval) \
  base::Ref<const C>(&data_.commitment), point, data_.eval

    for (size_t i = 0; i < data_.substitution_commitments.size(); ++i) {
      openings.emplace_back(
          OPENING(substitution_commitments[i], x, substitution_evals[i]));
    }
#undef OPENING
  }

 private:
  const F& GetEval(const ConstraintSystem<F>& constraint_system,
                   const AnyColumnKey& column) const {
    const absl::Span<const F>* evals = nullptr;
    switch (column.type()) {
      case ColumnType::kAdvice: {
        evals = &data_.advice_evals;
        break;
      }
      case ColumnType::kFixed: {
        evals = &data_.fixed_evals;
        break;
      }
      case ColumnType::kInstance: {
        evals = &data_.instance_evals;
        break;
      }
      case ColumnType::kAny: {
        NOTREACHED();
      }
    }
    return (
        *evals)[constraint_system.GetAnyQueryIndex(column, Rotation::Cur())];
  }

  const PermutationVerifierData<F, C>& data_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFIER_H_
