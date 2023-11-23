// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_SPARSE_EVALUATIONS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_SPARSE_EVALUATIONS_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/multivariate/support_poly_operators.h"

namespace tachyon::math {

template <typename F, std::size_t MaxDegree, std::size_t NumVars>
class MultilinearSparseEvaluations {
 public:
  static constexpr std::size_t kMaxDegree = MaxDegree;
  static constexpr std::size_t kNumVars = NumVars;

  using Field = F;

  constexpr explicit MultilinearSparseEvaluations()
      : evaluations_(), num_vars_(0), zero_(F::Zero()) {}

  constexpr explicit MultilinearSparseEvaluations(
      const std::vector<std::pair<std::size_t, F>>& evaluations)
      : evaluations_(evaluations.begin(), evaluations.end()),
        num_vars_(evaluations.size()),
        zero_(F::Zero()) {
    CHECK_LE(NumVars, MaxDegree);
  }

  constexpr explicit MultilinearSparseEvaluations(
      const std::vector<std::pair<std::size_t, F>>&& evaluations)
      : evaluations_(std::move(evaluations.begin()),
                     std::move(evaluations.end())),
        num_vars_(evaluations.size()),
        zero_(F::Zero()) {
    CHECK_LE(NumVars, MaxDegree);
  }

  constexpr static MultilinearSparseEvaluations Zero(std::size_t degree) {
    return MultilinearSparseEvaluations(
        std::vector<std::pair<std::size_t, F>>());
  }

  constexpr static MultilinearSparseEvaluations One(std::size_t degree) {
    std::vector<std::pair<std::size_t, F>> evaluations;
    for (std::size_t i = 0; i < (static_cast<std::size_t>(1) << degree); ++i) {
      evaluations.push_back({i, F::One()});
    }
    return MultilinearSparseEvaluations(evaluations);
  }

  constexpr static MultilinearSparseEvaluations Random(std::size_t num_vars,
                                                       absl::BitGen& bitgen) {
    return RandWithConfig(
        num_vars, static_cast<std::size_t>(1) << (num_vars / 2), bitgen);
  }

  constexpr bool operator==(const MultilinearSparseEvaluations& other) const {
    return evaluations_ == other.evaluations_;
  }

  constexpr bool operator!=(const MultilinearSparseEvaluations& other) const {
    return !operator==(other);
  }

  constexpr const F* Get(std::size_t i) const {
    auto it = evaluations_.find(i);
    if (it != evaluations_.end()) {
      return &(it->second);
    }
    return nullptr;
  }

  constexpr bool IsZero() const {
    return std::all_of(evaluations_.begin(), evaluations_.end(),
                       [](const std::pair<std::size_t, F>& pair) {
                         return pair.second.IsZero();
                       });
  }

  constexpr bool IsOne() const {
    return std::all_of(evaluations_.begin(), evaluations_.end(),
                       [](const std::pair<std::size_t, F>& pair) {
                         return pair.second.IsOne();
                       });
  }

  constexpr size_t Degree() const {
    if (evaluations_.empty()) {
      return 0;
    }

    size_t maxKey = 0;
    for (const auto& pair : evaluations_) {
      maxKey = std::max(maxKey, pair.first);
    }

    return base::bits::SafeLog2Ceiling(maxKey);
  }

  MultilinearSparseEvaluations FixVariables(
      const std::vector<F>& partial_point) const {
    int dim = partial_point.size();
    if (dim > num_vars_) {
      throw std::invalid_argument("invalid partial point dimension");
    }

    int window = base::bits::Log2Floor(evaluations_.size());
    std::vector<F> point = partial_point;
    absl::flat_hash_map<std::size_t, F> last = treemap_to_hashmap(evaluations_);

    while (!point.empty()) {
      int focus_length =
          (window > 0 && point.size() > window) ? window : point.size();
      std::vector<F> focus(point.begin(), point.begin() + focus_length);
      point.erase(point.begin(), point.begin() + focus_length);
      std::vector<F> pre = PrecomputeEq(focus);
      int dim = focus.size();
      absl::flat_hash_map<std::size_t, F> result;

      for (const auto& src_entry : last) {
        int old_idx = src_entry.first;
        F gz = pre[old_idx & ((1 << dim) - 1)];
        int new_idx = old_idx >> dim;
        result[new_idx] += gz * src_entry.second;
      }
      last = result;
    }

    MultilinearSparseEvaluations<F, MaxDegree, NumVars> result;
    result.num_vars_ = num_vars_ - dim;
    result.evaluations_ =
        absl::btree_map<std::size_t, F>(hashmap_to_treemap(last));
    result.zero_ = F::Zero();

    return result;
  }

  F Evaluate(const std::vector<F>& point) const {
    CHECK_LE(point.size(), num_vars_);

    MultilinearSparseEvaluations fixed = FixVariables(point);

    if (fixed.IsZero()) return F::Zero();
    return fixed.evaluations_.at(0);
  }

  const absl::btree_map<std::size_t, F> tuples_to_treemap(
      const std::vector<std::pair<std::size_t, F>>& tuples) const {
    absl::btree_map<std::size_t, F> result;
    for (const auto& entry : tuples) {
      result[entry.first] = entry.second;
    }
    return result;
  }

  const absl::flat_hash_map<std::size_t, F> treemap_to_hashmap(
      const absl::btree_map<std::size_t, F>& treemap) const {
    absl::flat_hash_map<std::size_t, F> hashmap;
    for (const auto& entry : treemap) {
      hashmap[entry.first] = entry.second;
    }
    return hashmap;
  }

  const absl::btree_map<std::size_t, F> hashmap_to_treemap(
      const absl::flat_hash_map<std::size_t, F>& map) const {
    absl::btree_map<std::size_t, F> tree_map;

    for (const auto& entry : map) {
      tree_map.insert({entry.first, entry.second});
    }

    return tree_map;
  }

  static MultilinearSparseEvaluations RandWithConfig(
      std::size_t num_vars, std::size_t num_nonzero_entries,
      absl::BitGen& bitgen) {
    assert(num_nonzero_entries <= (static_cast<std::size_t>(1) << num_vars));

    absl::flat_hash_map<std::size_t, F> map;

    for (std::size_t i = 0; i < num_nonzero_entries; ++i) {
      std::size_t index;
      do {
        index = absl::Uniform(bitgen, static_cast<std::size_t>(0),
                              static_cast<std::size_t>(1) << num_vars);
      } while (map.find(index) != map.end());
      map[index] = F::Random();
    }

    MultilinearSparseEvaluations result(
        std::vector<std::pair<std::size_t, F>>(map.begin(), map.end()));

    return result;
  }

  std::vector<F> PrecomputeEq(const std::vector<F>& g) const {
    int dim = g.size();
    std::vector<F> dp(1 << dim, F::Zero());
    dp[0] = F::One() - g[0];
    dp[1] = g[0];
    for (int i = 1; i < dim; ++i) {
      for (int b = 0; b < (1 << i); ++b) {
        F prev = dp[b];
        dp[b + (1 << i)] = prev * g[i];
        dp[b] = prev - dp[b + (1 << i)];
      }
    }
    return dp;
  }

  std::string ToString() const {
    std::string result = "[";
    bool firstEntry = true;

    for (const auto& entry : evaluations_) {
      if (!firstEntry) {
        result += ", ";
      }
      result += "(" + std::to_string(entry.first) + ", " +
                entry.second.ToString() + ")";
      firstEntry = false;
    }

    result += "]";
    return result;
  }

 private:
  friend class internal::MultilinearExtensionOp<
      MultilinearSparseEvaluations<F, MaxDegree, NumVars>>;

  absl::btree_map<std::size_t, F> evaluations_;
  std::size_t num_vars_;
  F zero_ = F::Zero();
};

}  // namespace tachyon::math
#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_SPARSE_EVALUATIONS_H_
