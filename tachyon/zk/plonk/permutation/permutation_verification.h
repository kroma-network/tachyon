// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/logging.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/permutation/permutation_utils.h"
#include "tachyon/zk/plonk/permutation/permutation_verification_data.h"

namespace tachyon::zk::plonk {

template <typename F, typename C>
const F& GetEval(const PermutationVerificationData<F, C>& data,
                 const ConstraintSystem<F>& constraint_system,
                 const AnyColumnKey& column) {
  const absl::Span<const F>* evals = nullptr;
  switch (column.type()) {
    case ColumnType::kAdvice: {
      evals = &data.advice_evals;
      break;
    }
    case ColumnType::kFixed: {
      evals = &data.fixed_evals;
      break;
    }
    case ColumnType::kInstance: {
      evals = &data.instance_evals;
      break;
    }
    case ColumnType::kAny: {
      NOTREACHED();
    }
  }
  return (*evals)[constraint_system.GetAnyQueryIndex(column, Rotation::Cur())];
}

template <typename F>
size_t GetSizeOfPermutationVerificationExpressions(
    const ConstraintSystem<F>& constraint_system) {
  size_t num_products = constraint_system.ComputePermutationProductNums();
  if (num_products == 0) return 0;
  return 2 * num_products + 1;
}

template <typename F, typename C>
std::vector<F> CreatePermutationVerificationExpressions(
    const PermutationVerificationData<F, C>& data,
    const ConstraintSystem<F>& constraint_system) {
  if (data.product_evals.empty()) return {};
  std::vector<F> ret;
  ret.reserve(GetSizeOfPermutationVerificationExpressions(constraint_system));
  // l_first(X) * (1 - z_first(X)) = 0
  const F& z_first = data.product_evals.front();
  ret.push_back(*data.l_first * (F::One() - z_first));
  // l_last(X) * (z_last(X)² - z_last(X)) = 0
  const F& z_last = data.product_evals.back();
  ret.push_back(*data.l_last * (z_last.Square() - z_last));
  // l_first(X) * (zᵢ(X) - zᵢ₋₁(ω^(last) * X)) = 0
  for (size_t i = 1; i < data.product_evals.size(); ++i) {
    const F& z_cur = data.product_evals[i];
    const F& z_prev_last = *data.product_last_evals[i - 1];
    ret.push_back(*data.l_first * (z_cur - z_prev_last));
  }
  // (1 - (l_last(X) + l_blind(X))) * (
  //   zᵢ(ω * X) Π (p(X) + β * sᵢ(X) + γ)
  // - zᵢ(X) Π (p(X) + δⁱ *β * X + γ)
  // ) = 0
  const PermutationArgument& argument = constraint_system.permutation();
  size_t chunk_idx = 0;
  const F& delta = GetDelta<F>();
  F active_rows = F::One() - (*data.l_last + *data.l_blind);
  size_t chunk_len = constraint_system.ComputePermutationChunkLen();
  for (absl::Span<const AnyColumnKey> columns :
       base::Chunked(argument.columns(), chunk_len)) {
    absl::Span<const F> common_evals = absl::MakeConstSpan(
        data.common_evals.data() + chunk_idx * chunk_len, columns.size());
    F left = data.product_next_evals[chunk_idx];
    for (size_t i = 0; i < columns.size(); ++i) {
      const AnyColumnKey& column = columns[i];
      const F& eval = GetEval(data, constraint_system, column);
      left *= eval + *data.beta * common_evals[i] + *data.gamma;
    }

    F right = data.product_evals[chunk_idx];
    F current_delta = *data.beta * *data.x * delta.Pow(chunk_idx * chunk_len);
    for (size_t i = 0; i < columns.size(); ++i) {
      const AnyColumnKey& column = columns[i];
      const F& eval = GetEval(data, constraint_system, column);
      right *= eval + current_delta + *data.gamma;
      current_delta *= delta;
    }
    ret.push_back(active_rows * (left - right));
    ++chunk_idx;
  }
  return ret;
}

template <typename F>
size_t GetSizeOfPermutationVerifierQueries(
    const ConstraintSystem<F>& constraint_system) {
  size_t num_products = constraint_system.ComputePermutationProductNums();
  size_t size = num_products * 2;
  if (num_products > 1) {
    size += num_products - 1;
  }
  return size;
}

template <typename PCS, typename F, typename C,
          typename Poly = typename PCS::Poly>
std::vector<crypto::PolynomialOpening<Poly, C>> CreatePermutationQueries(
    const PermutationVerificationData<F, C>& data,
    const ConstraintSystem<F>& constraint_system) {
  size_t num_products = constraint_system.ComputePermutationProductNums();
  std::vector<crypto::PolynomialOpening<Poly, C>> queries;
  queries.reserve(GetSizeOfPermutationVerifierQueries(constraint_system));
  // Open permutation product commitments at x and ω⁻¹ * x.
  // Open permutation product commitments at x and ω * x.
  for (size_t i = 0; i < num_products; ++i) {
    queries.emplace_back(base::Ref<const C>(&data.product_commitments[i]),
                         base::DeepRef<const F>(data.x), data.product_evals[i]);
    queries.emplace_back(base::Ref<const C>(&data.product_commitments[i]),
                         base::DeepRef<const F>(data.x_next),
                         data.product_next_evals[i]);
  }
  if (num_products > 1) {
    // Open it at ω^{last} * x for all but the last set.
    for (size_t i = num_products - 2; i != SIZE_MAX; --i) {
      queries.emplace_back(base::Ref<const C>(&data.product_commitments[i]),
                           base::DeepRef<const F>(data.x_last),
                           data.product_last_evals[i].value());
    }
  }
  return queries;
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_H_
