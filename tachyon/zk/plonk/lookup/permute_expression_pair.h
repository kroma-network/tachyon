// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_PERMUTE_EXPRESSION_PAIR_H_
#define TACHYON_ZK_PLONK_LOOKUP_PERMUTE_EXPRESSION_PAIR_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/base/prover.h"
#include "tachyon/zk/plonk/error.h"
#include "tachyon/zk/plonk/lookup/lookup_pair.h"

namespace tachyon::zk {

// Given a vector of input values A and a vector of table values S,
// this method permutes A and S to produce A' and S', such that:
// - like values in A' are vertically adjacent to each other; and
// - the first row in a sequence of like values in A' is the row
//   that has the corresponding value in S'.
// This method returns (A', S') if no errors are encountered.
template <typename PCSTy, typename Evals, typename F = typename Evals::Field>
Error PermuteExpressionPair(Prover<PCSTy>* prover, const LookupPair<Evals>& in,
                            LookupPair<Evals>* out) {
  size_t domain_size = prover->domain()->size();
  size_t blinding_factors = prover->blinder().blinding_factors();
  if (domain_size == 0) return Error::kConstraintSystemFailure;
  if (domain_size - 1 < blinding_factors)
    return Error::kConstraintSystemFailure;
  size_t usable_rows = domain_size - (blinding_factors + 1);

  std::vector<F> permuted_input_expressions = in.input().evaluations();

  // sort input lookup expression values
  std::sort(permuted_input_expressions.begin(),
            permuted_input_expressions.begin() + usable_rows);

  // a map of each unique element in the table expression and its count
  absl::btree_map<F, uint32_t> leftover_table_map;

  for (size_t i = 0; i < usable_rows; ++i) {
    const F& coeff = *in.table()[i];
    // if key doesn't exist, insert the key and value 1 for the key.
    auto it = leftover_table_map.try_emplace(coeff, uint32_t{1});
    // no inserted value, meaning the key exists.
    if (!it.second) {
      // Increase value by 1 if not inserted.
      ++((*it.first).second);
    }
  }

  std::vector<F> permuted_table_expressions =
      base::CreateVector(domain_size, F::Zero());

  std::vector<size_t> repeated_input_rows;
  for (size_t row = 0; row < usable_rows; ++row) {
    const F& input_value = permuted_input_expressions[row];

    // ref: https://zcash.github.io/halo2/design/proving-system/lookup.html
    //
    // Lookup Argument must satisfy these 2 constraints.
    //
    // - constraint 1: l₀(X) * (A'(X) - S'(x)) = 0
    // - constraint 2: (A'(X) - S'(x)) * (A'(X) - A'(ω⁻¹X)) = 0
    //
    // - What 'row == 0' condition means: l₀(x) == 1.
    // To satisfy constraint 1, A'(x) - S'(x) must be 0.
    // => checking if A'(x) == S'(x)
    // - What 'input_value != permuted_input_expressions[row-1]' condition
    //   means: (A'(x) - A'(ω⁻¹x)) != 0.
    // To satisfy constraint 2, A'(x) - S'(x) must be 0.
    // => checking if A'(x) == S'(x)
    //
    // Example
    //
    // Assume that
    //  * in.input.evaluations() = [1,2,1,5]
    //  * in.table.evaluations() = [1,2,4,5]
    //
    // Result after for loop
    //
    //                   A'                      S'
    //               --------                --------
    //              |    1   |              |    1   |
    //               --------                --------
    //              |    1   |              |    4   |
    //               --------                --------
    //              |    2   |              |    2   |
    //               --------                --------
    //              |    5   |              |    5   |
    //               --------                --------
    // we can see that elements of A' {1,2,5} is in S' {1,4,2,5}
    //
    if (row == 0 || input_value != permuted_input_expressions[row - 1]) {
      // Assign S'(x) with A'(x).
      permuted_table_expressions[row] = input_value;

      // remove one instance of input_value from |leftover_table_map|.
      auto it = leftover_table_map.find(input_value);
      // if input value is not found, return error
      if (it == leftover_table_map.end())
        return Error::kConstraintSystemFailure;

      // input value found, check if the value > 0.
      // then decrement the value by 1
      CHECK_GT(it->second--, size_t{0});
    } else {
      repeated_input_rows.push_back(row);
    }
  }

  // populate permuted table at unfilled rows with leftover table elements
  for (auto it = leftover_table_map.begin(); it != leftover_table_map.end();
       ++it) {
    const F& coeff = it->first;
    const uint32_t count = it->second;

    for (uint32_t i = 0; i < count; ++i) {
      CHECK(!repeated_input_rows.empty());
      size_t row = repeated_input_rows.back();
      permuted_table_expressions[row] = coeff;
      repeated_input_rows.pop_back();
    }
  }

  CHECK(repeated_input_rows.empty());

  Evals input(std::move(permuted_input_expressions));
  Evals table(std::move(permuted_table_expressions));

  prover->blinder().Blind(input);
  prover->blinder().Blind(table);

  *out = {std::move(input), std::move(table)};

  return Error::kNone;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_PERMUTE_EXPRESSION_PAIR_H_
