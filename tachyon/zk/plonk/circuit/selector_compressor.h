// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_SELECTOR_COMPRESSOR_H_
#define TACHYON_ZK_PLONK_CIRCUIT_SELECTOR_COMPRESSOR_H_

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/containers/cxx20_erase_vector.h"
#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/plonk/circuit/selector_assignment.h"
#include "tachyon/zk/plonk/circuit/selector_description.h"

namespace tachyon::zk {

template <typename F>
class SelectorCompressor {
 public:
  struct Result {
    std::vector<std::vector<F>> polys;
    std::vector<SelectorAssignment<F>> assignments;

    Result() = default;
    Result(std::vector<std::vector<F>>&& polys,
           std::vector<SelectorAssignment<F>>&& assignments)
        : polys(std::move(polys)), assignments(std::move(assignments)) {}
  };

  using AllocateFixedColumnCallback =
      base::RepeatingCallback<std::unique_ptr<Expression<F>>()>;

  // This function takes a vector that defines each selector as well as a
  // closure used to allocate new fixed columns, and returns the assignment of
  // each combination as well as details about each selector assignment.
  //
  // This function takes
  // * |selectors_in|, a vector of vector of bool.
  // * |max_degree|, the maximum allowed degree of any gate.
  // * |callback|, a closure that constructs a new fixed column and
  //   queries it at |Rotation::Cur()|, returning the expression.
  //
  // and returns |Result| containing the assignment of each new fixed
  // column (which each correspond to a combination) as well as a vector of
  // |SelectorAssignment<F>| that the caller can use to perform the necessary
  // substitutions to the constraint system.
  //
  // This function is completely deterministic.
  static Result Process(const std::vector<std::vector<bool>>& selectors_in,
                        const std::vector<size_t>& degrees, size_t max_degree,
                        AllocateFixedColumnCallback callback) {
    if (selectors_in.empty()) return {};

    // The length of all provided selectors must be the same.
    size_t n = selectors_in[0].size();
    DCHECK(std::all_of(selectors_in.begin(), selectors_in.end(),
                       [n](const std::vector<bool>& activations) {
                         return activations.size() == n;
                       }));

    std::vector<SelectorDescription> selectors =
        base::Map(selectors_in,
                  [&degrees](size_t i, const std::vector<bool>& activations) {
                    size_t max_degree = degrees[i];
                    return SelectorDescription(i, activations, max_degree);
                  });

    std::vector<std::vector<F>> combination_assignments;
    std::vector<SelectorAssignment<F>> selector_assignments;

    // All provided selectors of degree 0 are assumed to be either concrete
    // selectors or do not appear in a gate. Let's address these first.
    base::EraseIf(selectors, [&combination_assignments, &selector_assignments,
                              callback](const SelectorDescription& selector) {
      if (selector.max_degree() != 0) return false;
      // This is a complex selector, or a selector that does not
      // appear in any gate constraint.
      std::unique_ptr<Expression<F>> expression = callback.Run();

      std::vector<F> combination_assignment =
          base::Map(selector.activations(),
                    [](bool b) { return b ? F::One() : F::Zero(); });
      size_t combination_index = combination_assignments.size();
      combination_assignments.push_back(std::move(combination_assignment));
      selector_assignments.push_back(SelectorAssignment<F>(
          selector.selector_index(), combination_index, std::move(expression)));
      return true;
    });

    // All of the remaining |selectors| are simple. Let's try to combine them.
    // First, we compute the exclusion matrix that has (j, k) = true if selector
    // j and selector k conflict -- that is, they are both enabled on the same
    // row. This matrix is symmetric and the diagonal entries are false, so we
    // only need to store the lower triangular entries.
    std::vector<std::vector<bool>> exclusion_matrix = base::CreateVector(
        selectors.size(),
        [](size_t i) { return base::CreateVector(i, false); });

    for (size_t i = 0; i < selectors.size(); ++i) {
      const std::vector<bool>& rows = selectors[i].activations();
      for (size_t j = 0; j < i; ++j) {
        const SelectorDescription& other_selector = selectors[j];
        for (auto [l, r] : base::Zipped(rows, other_selector.activations())) {
          if (l & r) {
            // Mark them as incompatible
            exclusion_matrix[i][j] = true;
            break;
          }
        }
      }
    }

    // Simple selectors that we've added to combinations already.
    std::vector<bool> added = base::CreateVector(selectors.size(), false);
    for (size_t i = 0; i < selectors.size(); ++i) {
      if (added[i]) continue;
      added[i] = true;
      const SelectorDescription& selector = selectors[i];
      CHECK_LE(selector.max_degree(), max_degree);
      // This is used to keep track of the largest degree gate involved in the
      // combination so far. We subtract by one to omit the virtual selector
      // which will be substituted by the caller with the expression we give
      // them.
      size_t d = selector.max_degree() - 1;
      std::vector<SelectorDescription> combination = {selector};
      std::vector<size_t> combination_added = {i};

      // Try to find other selectors that can join this one.
      for (size_t j = i + 1; j < selectors.size(); ++j) {
        if (d + combination.size() == max_degree) {
          // Short circuit; nothing can be added to this
          // combination.
          break;
        }

        // Skip selectors that have been added to previous combinations
        if (added[j]) continue;

        // Is this selector excluded from co-existing in the same
        // combination with any of the other selectors so far?
        bool excluded = false;
        for (size_t i : combination_added) {
          if (exclusion_matrix[j][i]) {
            excluded = true;
          }
        }
        if (excluded) continue;

        // Can the new selector join the combination? Reminder: we use
        // |selector.max_degree() - 1| to omit the influence of the virtual
        // selector on the degree, as it will be substituted.
        size_t new_d = std::max(d, selector.max_degree() - 1);
        if (new_d + combination.size() + 1 > max_degree) {
          // Guess not.
          continue;
        }

        d = new_d;
        combination.push_back(selector);
        combination_added.push_back(j);
        added[j] = true;
      }

      // Now, compute the selector and combination assignments.
      std::vector<F> combination_assignment = base::CreateVector(n, F::Zero());
      size_t combination_len = combination.size();
      size_t combination_index = combination_assignments.size();
      std::unique_ptr<Expression<F>> query = callback.Run();

      F assigned_root = F::One();
      selector_assignments.reserve(selector_assignments.size() +
                                   combination.size());
      for (const SelectorDescription& selector : combination) {
        // Compute the expression for substitution. This produces an expression
        // of the form
        //     q * Prod[i = 1..=combination_len, i != assigned_root](i - q)
        //
        // which is non-zero only on rows where |combination_assignment| is set
        // to |assigned_root|. In particular, rows set to 0 correspond to all
        // selectors being disabled.
        std::unique_ptr<Expression<F>> expression = query->Clone();
        F root = F::One();
        for (size_t i = 0; i < combination_len; ++i) {
          if (root != assigned_root) {
            expression = std::move(expression) *
                         (ExpressionFactory<F>::Constant(root) - query);
          }
          root += F::One();
        }

        // Update the combination assignment
        const std::vector<bool>& activations = selector.activations();
        for (size_t k = 0; k < n; ++k) {
          // This will not overwrite another selector's activations
          // because we have ensured that selectors are disjoint.
          if (activations[k]) {
            combination_assignment[k] = assigned_root;
          }
        }

        assigned_root += F::One();
        selector_assignments.emplace_back(selector.selector_index(),
                                          combination_index,
                                          std::move(expression));
      }
      combination_assignments.push_back(std::move(combination_assignment));
    }

    return {
        std::move(combination_assignments),
        std::move(selector_assignments),
    };
  }
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_SELECTOR_COMPRESSOR_H_
