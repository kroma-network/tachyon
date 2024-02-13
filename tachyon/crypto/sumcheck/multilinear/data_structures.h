#ifndef TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_DATA_STRUCTURES_H_
#define TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_DATA_STRUCTURES_H_

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/container/node_hash_map.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/polynomials/multivariate/multilinear_extension.h"

namespace tachyon {

namespace crypto::sumcheck {

// This data structure  will be used as the verifier key.
struct TACHYON_EXPORT PolynomialInfo {
  size_t max_multiplicands;
  size_t num_variables;

  bool operator==(const PolynomialInfo& other) const {
    return max_multiplicands == other.max_multiplicands &&
           num_variables == other.num_variables;
  }
  bool operator!=(const PolynomialInfo& other) const {
    return !operator==(other);
  }
  // TODO(ashjeong): For better unit testing
  // PolynomialInfo Random() {}
};

template <typename Evaluations>
class ListOfProductsOfPolynomials;

template <typename Evaluations>
class ListOfProductsOfPolynomials<math::MultilinearExtension<Evaluations>> {
 public:
  using F = typename math::MultilinearExtension<Evaluations>::Field;

  struct Term {
    F coefficient;
    std::vector<size_t> indexes;
  };

  ListOfProductsOfPolynomials() = default;
  explicit ListOfProductsOfPolynomials(size_t num_variables)
      : num_variables_(num_variables) {}
  ListOfProductsOfPolynomials(ListOfProductsOfPolynomials& other) = default;
  ListOfProductsOfPolynomials& operator=(ListOfProductsOfPolynomials& other) =
      default;

  // Adds a Term to the list
  void AddTerm(
      F& coefficient,
      std::vector<std::shared_ptr<math::MultilinearExtension<Evaluations>>>&
          evaluations) {
    CHECK(!evaluations.empty());
    size_t num_evaluations = evaluations.size();
    max_evaluations_ = std::max(max_evaluations_, num_evaluations);
    std::vector<size_t> indexed_evaluations;
    indexed_evaluations.reserve(num_evaluations);
    for (std::shared_ptr<math::MultilinearExtension<Evaluations>> evaluation :
         evaluations) {
      CHECK_EQ(evaluation->Degree(), num_variables_);
      const math::MultilinearExtension<Evaluations>* evaluation_ptr =
          evaluation.get();
      auto it = raw_pointers_lookup_table_.find(evaluation_ptr);
      if (it != raw_pointers_lookup_table_.end()) {
        indexed_evaluations.push_back(
            raw_pointers_lookup_table_[evaluation_ptr]);
      } else {
        size_t curr_index = flattened_ml_evaluations_.size();
        flattened_ml_evaluations_.push_back(evaluation);
        raw_pointers_lookup_table_[evaluation_ptr] = curr_index;
        indexed_evaluations.push_back(curr_index);
      }
    }
    terms_.push_back(Term{coefficient, indexed_evaluations});
  }

  // Evaluates all terms together on a an evaluation point for each variable:
  // $$\sum_{i=0}^{n}coefficient_i\cdot\prod_{j=0}^{m_i}evaluation_{ij}$$
  F Evaluate(std::vector<F>& point) const {
    CHECK_EQ(point.size(), num_variables_);
    return std::accumulate(
        terms_.begin(), terms_.end(), F::Zero(),
        [this, &point](F& sum, const Term& term) {
          return sum +=
                 term.coefficient *
                 std::accumulate(
                     term.indexes.begin(), term.indexes.end(), F::One(),
                     [this, &point](F& product, size_t index) {
                       return product *=
                              this->flattened_ml_evaluations_[index]->Evaluate(
                                  point);
                     });
        });
  }

  // Extract the max number of evaluations and number of variables
  PolynomialInfo Info() const { return {max_evaluations_, num_variables_}; }

 private:
  // max number of evaluations for each term
  size_t max_evaluations_ = 0;
  // number of variables of the polynomial
  // Note(ashjeong): this is equivalent to `kMaxDegree` of multilinear
  // extensions
  size_t num_variables_ = 0;
  // list of coefficients and their corresponding list of indexes in reference
  // to evaluations in flattened_ml_evaluations_
  std::vector<Term> terms_;
  // holds shared pointers of multilinear evaluations
  std::vector<std::shared_ptr<math::MultilinearExtension<Evaluations>>>
      flattened_ml_evaluations_;
  // holds all unique dense ML evaluations currently used + their corresponding
  // index in "flattened_ml_evaluations_"
  // not owned
  absl::node_hash_map<const math::MultilinearExtension<Evaluations>*, size_t>
      raw_pointers_lookup_table_;
};

// NOTE(ashjeong): univariate_evaluations.h used as reference
template <typename H, typename F, typename Evaluations>
H AbslHashValue(H h, const math::MultilinearExtension<Evaluations>* evals) {
  size_t degree = 0;
  for (const F& eval : evals->evaluations()) {
    h = H::combine(std::move(h), eval);
    ++degree;
  }
  F zero = F::Zero();
  size_t max_degree = evals->Degree();
  for (size_t i = degree; i <= max_degree; ++i) {
    h = H::combine(std::move(h), zero);
  }
  return h;
}

}  // namespace crypto::sumcheck

namespace base {

using PolynomialInfo = crypto::sumcheck::PolynomialInfo;

// Implements serialization/deserialization for PolynomialInfo
template <>
class Copyable<PolynomialInfo> {
 public:
  static bool WriteTo(const PolynomialInfo& bigint, Buffer* buffer) {
    return buffer->WriteMany(bigint.max_multiplicands, bigint.num_variables);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, PolynomialInfo* bigint) {
    size_t max_multiplicands;
    size_t num_variables;
    if (!buffer.ReadMany(&max_multiplicands, &num_variables)) return false;
    *bigint = {max_multiplicands, num_variables};
    return true;
  }

  static size_t EstimateSize(const PolynomialInfo& bigint) {
    return base::EstimateSize(bigint.max_multiplicands) +
           base::EstimateSize(bigint.num_variables);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_SUMCHECK_MULTILINEAR_DATA_STRUCTURES_H_
