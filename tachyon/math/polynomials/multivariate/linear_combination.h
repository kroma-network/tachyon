// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_LINEAR_COMBINATION_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_LINEAR_COMBINATION_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/random.h"
#include "tachyon/math/polynomials/multivariate/linear_combination_term.h"

namespace tachyon::math {

template <typename MLE>
class LinearCombination {
 public:
  using F = typename MLE::Field;
  using Point = typename MLE::Point;

  LinearCombination() = default;
  explicit LinearCombination(size_t num_variables)
      : num_variables_(num_variables) {}
  LinearCombination(LinearCombination& other) = default;
  LinearCombination& operator=(LinearCombination& other) = default;

  size_t max_evaluations() const { return max_evaluations_; }
  size_t num_variables() const { return num_variables_; }
  const std::vector<LinearCombinationTerm<F>>& terms() const { return terms_; }
  const std::vector<std::shared_ptr<MLE>>& flattened_ml_evaluations() const {
    return flattened_ml_evaluations_;
  }

  static LinearCombination Random(size_t num_variables,
                                  base::Range<size_t> max_evaluations_range,
                                  size_t num_terms) {
    LinearCombination linear_combination(num_variables);

    for (size_t i = 0; i < num_terms; ++i) {
      F coefficient = F::Random();
      size_t max_evaluations = base::Uniform(max_evaluations_range);
      std::vector<std::shared_ptr<MLE>> evaluations =
          base::CreateVector(max_evaluations, [num_variables]() {
            return std::make_shared<MLE>(MLE::Random(num_variables));
          });
      linear_combination.AddTerm(coefficient, evaluations);
    }
    return linear_combination;
  }

  // Adds a |LinearCombinationTerm| to member |terms_|
  void AddTerm(const F& coefficient,
               const std::vector<std::shared_ptr<MLE>>& evaluations) {
    if (evaluations.empty()) return;
    size_t num_evaluations = evaluations.size();
    max_evaluations_ = std::max(max_evaluations_, num_evaluations);
    std::vector<size_t> indexes =
        base::Map(evaluations, [this](const std::shared_ptr<MLE>& evaluation) {
          CHECK_EQ(evaluation->Degree(), num_variables_);
          const MLE* evaluation_ptr = evaluation.get();
          auto it = lookup_table_.try_emplace(evaluation_ptr,
                                              flattened_ml_evaluations_.size());
          // if inserted
          if (it.second) {
            flattened_ml_evaluations_.push_back(evaluation);
          }
          return (*it.first).second;
        });
    terms_.push_back({coefficient, std::move(indexes)});
  }

  // Combines the given |LinearCombination| into “H”, or the sum of all possible
  // values of all the variables. Refer to
  // https://people.cs.georgetown.edu/jthaler/sumcheck.pdf for more info.
  F Combine() const {
    std::vector<F> results = base::ParallelizeMap(
        terms_, [this](absl::Span<const LinearCombinationTerm<F>> chunk) {
          return CombineSerial(num_variables_, flattened_ml_evaluations_,
                               chunk);
        });
    return std::accumulate(results.begin(), results.end(), F::Zero(),
                           std::plus<>());
  }

  // Evaluates all terms together on an evaluation point for each variable:
  // ∑ᵢ₌₀..ₙ(coefficientᵢ⋅∏ⱼ₌₀..ₘevaluationᵢⱼ)
  // where n = total terms, m = number of evaluations per term
  F Evaluate(const Point& point) const {
    CHECK_EQ(point.size(), num_variables_);
    std::vector<F> results = base::ParallelizeMap(
        terms_,
        [this, &point](absl::Span<const LinearCombinationTerm<F>> chunk) {
          return EvaluateSerial(point, flattened_ml_evaluations_, chunk);
        });
    return std::accumulate(results.begin(), results.end(), F::Zero(),
                           std::plus<>());
  }

 private:
  static F CombineSerial(
      size_t num_variables,
      const std::vector<std::shared_ptr<MLE>>& flattened_ml_evaluations,
      absl::Span<const LinearCombinationTerm<F>> terms) {
    return std::accumulate(
        terms.begin(), terms.end(), F::Zero(),
        [num_variables, &flattened_ml_evaluations](
            F& acc, const LinearCombinationTerm<F>& term) {
          return acc += term.Combine(num_variables, flattened_ml_evaluations);
        });
  }

  static F EvaluateSerial(
      const Point& point,
      const std::vector<std::shared_ptr<MLE>>& flattened_ml_evaluations,
      absl::Span<const LinearCombinationTerm<F>> terms) {
    return std::accumulate(terms.begin(), terms.end(), F::Zero(),
                           [&point, &flattened_ml_evaluations](
                               F& acc, const LinearCombinationTerm<F>& term) {
                             return acc += term.Evaluate(
                                        point, flattened_ml_evaluations);
                           });
  }

  // max number of evaluations for each term
  size_t max_evaluations_ = 0;
  // number of variables of the polynomial
  size_t num_variables_ = 0;
  // list of coefficients and their corresponding list of indexes in reference
  // to evaluations in |flattened_ml_evaluations_|
  std::vector<LinearCombinationTerm<F>> terms_;
  // holds shared pointers of unique multilinear evaluations
  std::vector<std::shared_ptr<MLE>> flattened_ml_evaluations_;
  // holds all unique dense multilinear evaluations currently used & their
  // corresponding index in |flattened_ml_evaluations_|
  // not owned
  absl::flat_hash_map<const MLE*, size_t> lookup_table_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_LINEAR_COMBINATION_H_
