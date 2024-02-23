// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_COMPRESSOR_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_COMPRESSOR_H_

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/containers/cxx20_erase_vector.h"
#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/plonk/constraint_system/exclusion_matrix.h"
#include "tachyon/zk/plonk/constraint_system/selector_assignment.h"
#include "tachyon/zk/plonk/constraint_system/selector_description.h"

namespace tachyon::zk::plonk {

template <typename F>
class SelectorCompressor {
 public:
  using AllocateFixedColumnCallback =
      base::RepeatingCallback<std::unique_ptr<Expression<F>>()>;

  SelectorCompressor() = default;

  const std::vector<std::vector<F>>& combination_assignments() const {
    return combination_assignments_;
  }
  const std::vector<SelectorAssignment<F>>& selector_assignments() const {
    return selector_assignments_;
  }

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
  void Process(const std::vector<std::vector<bool>>& selectors_in,
               const std::vector<size_t>& degrees, size_t max_degree,
               AllocateFixedColumnCallback callback) {
    if (selectors_in.empty()) return;

    callback_ = std::move(callback);

    // For example, suppose we have the following inputs:
    //
    // |selectors_in| = [
    //   [1, 0, 0, 0, 0, 0, 0, 0, 1],
    //   [1, 0, 0, 0, 0, 0, 0, 1, 0],
    //   [1, 0, 0, 0, 0, 0, 1, 0, 0],
    //   [0, 1, 0, 0, 0, 1, 1, 1, 0],
    //   [0, 1, 0, 0, 1, 0, 1, 0, 1],
    //   [0, 1, 0, 1, 0, 0, 0, 1, 1],
    //   [0, 0, 1, 1, 1, 0, 0, 0, 0],
    //   [0, 0, 1, 1, 0, 1, 0, 0, 0],
    //   [0, 0, 1, 0, 1, 1, 0, 0, 0],
    //   [1, 1, 1, 1, 1, 1, 1, 1, 1],
    // ]
    //
    // |degrees| = [3, 3, 3, 3, 3, 3, 7, 7, 7, 0]
    //
    // |max_degree| = 10
    //
    // The length of all provided selectors must be the same.
    size_t n = selectors_in[0].size();
    DCHECK(std::all_of(selectors_in.begin(), selectors_in.end(),
                       [n](const std::vector<bool>& activations) {
                         return activations.size() == n;
                       }));

    // |selectors| = [
    //   s₀: SelectorDescription(0, [1, 0, 0, 0, 0, 0, 0, 0, 1], 3),
    //   s₁: SelectorDescription(1, [1, 0, 0, 0, 0, 0, 0, 1, 0], 3),
    //   s₂: SelectorDescription(2, [1, 0, 0, 0, 0, 0, 1, 0, 0], 3),
    //   s₃: SelectorDescription(3, [0, 1, 0, 0, 0, 1, 1, 1, 0], 3),
    //   s₄: SelectorDescription(4, [0, 1, 0, 0, 1, 0, 1, 0, 1], 3),
    //   s₅: SelectorDescription(5, [0, 1, 0, 1, 0, 0, 0, 1, 1], 3),
    //   s₆: SelectorDescription(6, [0, 0, 1, 1, 1, 0, 0, 0, 0], 7),
    //   s₇: SelectorDescription(7, [0, 0, 1, 1, 0, 1, 0, 0, 0], 7),
    //   s₈: SelectorDescription(8, [0, 0, 1, 0, 1, 1, 0, 0, 0], 7),
    //   s₉: SelectorDescription(9, [1, 1, 1, 1, 1, 1, 1, 1, 1], 0),
    // ]
    selectors_ =
        base::Map(selectors_in,
                  [&degrees](size_t i, const std::vector<bool>& activations) {
                    size_t max_degree = degrees[i];
                    return SelectorDescription(i, &activations, max_degree);
                  });

    // Selectors with zero degree should be handled first before combining.
    HandleZeroDegreeSelectors();

    // All of the remaining |selectors| are simple. Let's try to combine them.
    // First, we compute the exclusion matrix.
    // See tachyon/zk/plonk/constraint_system/exclusion_matrix.h for details.
    ExclusionMatrix exclusion_matrix(selectors_);

    // Then, we combine the remaining |selectors|.
    // combination for s₀, s₃, s₆:
    //   s₀: [1, 0, 0, 0, 0, 0, 0, 0, 1] -> [1, 0, 0, 0, 0, 0, 0, 0, 1]
    //   s₃: [0, 1, 0, 0, 0, 1, 1, 1, 0] -> [1, 2, 0, 0, 0, 2, 2, 2, 1]
    //   s₆: [0, 0, 1, 1, 1, 0, 0, 0, 0] -> [1, 2, 3, 3, 3, 2, 2, 2, 1]
    //   => [1, 2, 3, 3, 3, 2, 2, 2, 1]
    //
    // combination for s₁, s₄, s₇:
    //   s₁: [1, 0, 0, 0, 0, 0, 0, 1, 0] -> [1, 0, 0, 0, 0, 0, 0, 1, 0]
    //   s₄: [0, 1, 0, 0, 1, 0, 1, 0, 1] -> [1, 2, 0, 0, 2, 0, 2, 1, 2]
    //   s₇: [0, 0, 1, 1, 0, 1, 0, 0, 0] -> [1, 2, 3, 3, 2, 3, 2, 1, 2]
    //   => [1, 2, 3, 3, 2, 3, 2, 1, 2]
    //
    // combination for s₂, s₅, s₈:
    //   s₂: [1, 0, 0, 0, 0, 0, 1, 0, 0] -> [1, 0, 0, 0, 0, 0, 1, 0, 0]
    //   s₅: [0, 1, 0, 1, 0, 0, 0, 1, 1] -> [1, 2, 0, 2, 0, 0, 1, 2, 2]
    //   s₈: [0, 0, 1, 0, 1, 1, 0, 0, 0] -> [1, 2, 3, 2, 3, 3, 1, 2, 2]
    //   => [1, 2, 3, 2, 3, 3, 1, 2, 2]
    //
    // |combination_assignments| = [
    //   [1, 1, 1, 1, 1, 1, 1, 1, 1],
    //   [1, 2, 3, 3, 3, 2, 2, 2, 1],
    //   [1, 2, 3, 3, 2, 3, 2, 1, 2],
    //   [1, 2, 3, 2, 3, 3, 1, 2, 2],
    // ]
    CombineSimpleSelectors(n, exclusion_matrix, max_degree);
  }

 private:
  FRIEND_TEST(SelectorCompressorTest, HandleZeroDegreeSelectors);
  FRIEND_TEST(SelectorCompressorTest, ConstructCombinedSelector);

  void HandleZeroDegreeSelectors() {
    // All provided selectors of degree 0 are assumed to be either concrete
    // selectors or do not appear in a gate. Let's address these first.
    // s₉ should be erased.
    base::EraseIf(selectors_, [this](const SelectorDescription& selector) {
      if (selector.max_degree() != 0) return false;
      // This is a complex selector, or a selector that does not
      // appear in any gate constraint.
      std::unique_ptr<Expression<F>> expression = callback_.Run();

      std::vector<F> combination_assignment =
          base::Map(selector.activations(),
                    [](bool b) { return b ? F::One() : F::Zero(); });
      size_t combination_index = combination_assignments_.size();
      combination_assignments_.push_back(std::move(combination_assignment));
      selector_assignments_.push_back(SelectorAssignment<F>(
          selector.selector_index(), combination_index, std::move(expression)));
      return true;
    });
  }

  void CombineSimpleSelectors(size_t n, const ExclusionMatrix& exclusion_matrix,
                              size_t max_degree) {
    // Simple selectors that we've added to combinations already.
    std::vector<bool> added(selectors_.size(), false);
    // For example in the first iteration, it adds:
    //   s₀: SelectorDescription(0, [1, 0, 0, 0, 0, 0, 0, 0, 1], 3)
    //   s₃: SelectorDescription(3, [0, 1, 0, 0, 0, 1, 1, 1, 0], 3)
    //   s₆: SelectorDescription(6, [0, 0, 1, 1, 1, 0, 0, 0, 0], 7)
    for (size_t i = 0; i < selectors_.size(); ++i) {
      if (added[i]) continue;
      added[i] = true;
      const SelectorDescription& selector = selectors_[i];
      CHECK_LE(selector.max_degree(), max_degree);
      // This is used to keep track of the largest degree gate involved in the
      // combination so far. We subtract by one to omit the virtual selector
      // which will be substituted by the caller with the expression we give
      // them.
      size_t d = selector.max_degree() - 1;
      std::vector<SelectorDescription> combination = {selector};
      std::vector<size_t> combination_added = {i};

      // Try to find other selectors that can join this one.
      for (size_t j = i + 1; j < selectors_.size(); ++j) {
        // The iteration breaks after s₆ is added to the combination.
        // |s₆.max_degree()| - 1 + |combination.size()| = 7 + 3 = 10
        if (d + combination.size() == max_degree) {
          // Short circuit; nothing can be added to this
          // combination.
          break;
        }

        // Skip selectors that have been added to previous combinations
        if (added[j]) continue;

        // Is this selector excluded from co-existing in the same
        // combination with any of the other selectors so far?
        // s₁, s₂, s₄, s₅, s₇, s₈ are excluded in the first iteration.
        bool excluded =
            std::any_of(combination_added.begin(), combination_added.end(),
                        [&exclusion_matrix, j](size_t i) {
                          return exclusion_matrix.IsExclusive(j, i);
                        });
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
        combination.push_back(selectors_[j]);
        combination_added.push_back(j);
        added[j] = true;
      }

      ConstructCombinedSelector(n, combination);
    }
  }

  void ConstructCombinedSelector(
      size_t n, const std::vector<SelectorDescription>& combination) {
    // Now, compute the selector and combination assignments.
    std::vector<F> combination_assignment(n);
    size_t combination_len = combination.size();
    size_t combination_index = combination_assignments_.size();
    std::unique_ptr<Expression<F>> query = callback_.Run();

    F assigned_root = F::One();
    selector_assignments_.reserve(selector_assignments_.size() +
                                  combination.size());
    for (const SelectorDescription& selector : combination) {
      // Compute the expression for substitution. This produces an expression
      // of the form
      //     q * Prod[i = 1..combination_len, i != assigned_root](i - q)
      //
      // which is non-zero only on rows where |combination_assignment| is set
      // to |assigned_root|. In particular, rows set to 0 correspond to all
      // selectors being disabled.
      // For example, in the first iteration, it produces:
      //     q * (2 - q) * (3 - q)
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
      for (size_t i = 0; i < n; ++i) {
        // This will not overwrite another selector's activations
        // because we have ensured that selectors are disjoint.
        if (activations[i]) {
          combination_assignment[i] = assigned_root;
        }
      }

      assigned_root += F::One();
      selector_assignments_.emplace_back(
          selector.selector_index(), combination_index, std::move(expression));
    }
    combination_assignments_.push_back(std::move(combination_assignment));
  }

  std::vector<std::vector<F>> combination_assignments_;
  std::vector<SelectorAssignment<F>> selector_assignments_;
  AllocateFixedColumnCallback callback_;
  std::vector<SelectorDescription> selectors_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_COMPRESSOR_H_
