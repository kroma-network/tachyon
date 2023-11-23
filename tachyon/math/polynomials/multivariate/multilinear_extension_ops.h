#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_

#include <algorithm>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/math/polynomials/multivariate/multilinear_extension.h"

namespace tachyon::math {
namespace internal {

template <typename F, size_t MaxDegree>
class MultilinearExtensionOp<MultilinearDenseEvaluations<F, MaxDegree>> {
 public:
  using D = MultilinearDenseEvaluations<F, MaxDegree>;
  using S = MultilinearSparseEvaluations<F, MaxDegree, MaxDegree>;

  static MultilinearExtension<D>& AddInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] += r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D>& SubInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] -= r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D>& NegInPlace(MultilinearExtension<D>& self) {
    std::vector<F>& evaluations = self.evaluations_.evaluations_;
    // clang-format off
    OPENMP_PARALLEL_FOR(F& evaluation : evaluations) {
      // clang-format on
      evaluation.NegInPlace();
    }
    return self;
  }

  static MultilinearExtension<D>& MulInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      l_evaluations[i] *= r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D>& DivInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      l_evaluations[i] /= r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<S> ToSparse(const MultilinearExtension<D>& self) {
    const std::vector<F>& dense_evaluations = self.evaluations_.evaluations_;
    // std::size_t num_vars =
    // MultilinearDenseEvaluations<F, MaxDegree>::kMaxDegree;

    absl::btree_map<std::size_t, F> sparse_evaluations;

    for (std::size_t index = 0; index < dense_evaluations.size(); ++index) {
      if (dense_evaluations[index] != F::Zero()) {
        sparse_evaluations[index] = dense_evaluations[index];
      }
    }
    std::vector<std::pair<std::size_t, F>> sparse_evaluations_vector;

    for (const auto& pair : sparse_evaluations) {
      sparse_evaluations_vector.push_back(pair);
    }

    return MultilinearExtension<S>(S(sparse_evaluations_vector));
  }
};

template <typename F, size_t MaxDegree, size_t NumVars>
class MultilinearExtensionOp<
    MultilinearSparseEvaluations<F, MaxDegree, NumVars>> {
 public:
  using D = MultilinearDenseEvaluations<F, MaxDegree>;
  using S = MultilinearSparseEvaluations<F, MaxDegree, NumVars>;

  static MultilinearExtension<S>& AddInPlace(
      MultilinearExtension<S>& self, const MultilinearExtension<S>& rhs) {
    if (self.IsZero()) {
      self = rhs;
      return self;
    }
    if (rhs.IsZero()) {
      return self;
    }

    assert(
        self.evaluations_.num_vars_ == rhs.evaluations_.num_vars_ &&
        "trying to add non-zero polynomial with different number of variables");

    absl::btree_map<std::size_t, F> evaluations;
    for (const auto& pair : self.evaluations_.evaluations_) {
      evaluations[pair.first] += pair.second;
    }
    for (const auto& pair : rhs.evaluations_.evaluations_) {
      evaluations[pair.first] += pair.second;
    }

    std::vector<std::pair<int, F>> non_zero_evaluations;
    for (const auto& pair : evaluations) {
      if (pair.second != F::Zero()) {
        non_zero_evaluations.emplace_back(pair.first, pair.second);
      }
    }

    self.evaluations_.evaluations_ = absl::btree_map<std::size_t, F>(
        non_zero_evaluations.begin(), non_zero_evaluations.end());
    self.evaluations_.num_vars_ = self.evaluations_.num_vars_;
    self.evaluations_.zero_ = F::Zero();

    return self;
  }

  static MultilinearExtension<S>& SubInPlace(
      MultilinearExtension<S>& self, const MultilinearExtension<S>& rhs) {
    if (self.IsZero()) {
      self = -rhs;
      return self;
    }
    if (rhs.IsZero()) {
      return self;
    }

    assert(
        self.evaluations_.num_vars_ == rhs.evaluations_.num_vars_ &&
        "trying to add non-zero polynomial with different number of variables");

    absl::btree_map<std::size_t, F> evaluations;
    for (const auto& pair : self.evaluations_.evaluations_) {
      evaluations[pair.first] += pair.second;
    }
    for (const auto& pair : rhs.evaluations_.evaluations_) {
      evaluations[pair.first] -= pair.second;
    }

    std::vector<std::pair<int, F>> non_zero_evaluations;
    for (const auto& pair : evaluations) {
      if (pair.second != F::Zero()) {
        non_zero_evaluations.emplace_back(pair.first, pair.second);
      }
    }

    self.evaluations_.evaluations_ = absl::btree_map<std::size_t, F>(
        non_zero_evaluations.begin(), non_zero_evaluations.end());
    self.evaluations_.num_vars_ = self.evaluations_.num_vars_;
    self.evaluations_.zero_ = F::Zero();

    return self;
  }

  static MultilinearExtension<S>& NegInPlace(MultilinearExtension<S>& self) {
    absl::btree_map<std::size_t, F>& evaluations =
        self.evaluations_.evaluations_;
    // clang-format off
   #pragma omp parallel for
  for (auto& pair : evaluations) {
    pair.second.NegInPlace();
  }
    // clang-format on
    return self;
  }

  static MultilinearExtension<S>& MulInPlace(
      MultilinearExtension<S>& self, const MultilinearExtension<S>& rhs) {
    if (self.IsZero() || rhs.IsZero()) {
      self = MultilinearExtension<S>::Zero(0);
      return self;
    }

    assert(
        self.evaluations_.num_vars_ == rhs.evaluations_.num_vars_ &&
        "trying to add non-zero polynomial with different number of variables");

    absl::btree_map<std::size_t, F> evaluations;
    for (const auto& self_pair : self.evaluations_.evaluations_) {
      for (const auto& rhs_pair : rhs.evaluations_.evaluations_) {
        if (self_pair.first == rhs_pair.first) {
          F product = self_pair.second * rhs_pair.second;
          evaluations[self_pair.first] += product;
        }
      }
    }

    std::vector<std::pair<size_t, F>> non_zero_evaluations;
    for (const auto& pair : evaluations) {
      if (pair.second != F::Zero()) {
        non_zero_evaluations.emplace_back(pair.first, pair.second);
      }
    }

    self.evaluations_.evaluations_ = absl::btree_map<std::size_t, F>(
        non_zero_evaluations.begin(), non_zero_evaluations.end());
    self.evaluations_.num_vars_ = self.evaluations_.num_vars_;
    self.evaluations_.zero_ = F::Zero();

    return self;
  }

  static MultilinearExtension<S>& DivInPlace(
      MultilinearExtension<S>& self, const MultilinearExtension<S>& rhs) {
    if (rhs.IsZero()) {
      throw std::runtime_error("Division by zero encountered");
    }

    assert(
        self.evaluations_.num_vars_ == rhs.evaluations_.num_vars_ &&
        "trying to add non-zero polynomial with different number of variables");
    absl::btree_map<std::size_t, F> evaluations;
    for (const auto& self_pair : self.evaluations_.evaluations_) {
      auto rhs_it = rhs.evaluations_.evaluations_.find(self_pair.first);
      if (rhs_it != rhs.evaluations_.evaluations_.end() &&
          rhs_it->second != F::Zero()) {
        F div = self_pair.second / rhs_it->second;
        evaluations[self_pair.first] += div;
      } else {
        throw std::runtime_error("Division by zero encountered");
      }
    }

    std::vector<std::pair<size_t, F>> non_zero_evaluations;
    for (const auto& pair : evaluations) {
      if (pair.second != F::Zero()) {
        non_zero_evaluations.emplace_back(pair.first, pair.second);
      }
    }

    self.evaluations_.evaluations_ = absl::btree_map<std::size_t, F>(
        non_zero_evaluations.begin(), non_zero_evaluations.end());
    self.evaluations_.num_vars_ = self.evaluations_.num_vars_;
    self.evaluations_.zero_ = F::Zero();

    return self;
  }

  static MultilinearExtension<D> ToDense(const MultilinearExtension<S>& self) {
    const absl::btree_map<std::size_t, F>& sparse_evaluations =
        self.evaluations_.evaluations_;
    std::size_t num_vars =
        MultilinearSparseEvaluations<F, MaxDegree, NumVars>::kNumVars;

    std::vector<F> dense_evaluations(1 << num_vars, F::Zero());

    for (const auto& entry : sparse_evaluations) {
      std::size_t index = entry.first;
      if (index < (1 << num_vars)) {
        dense_evaluations[index] = entry.second;
      }
    }

    return MultilinearExtension<D>(D(dense_evaluations));
  }

  static MultilinearExtension<S> ToSparse(const MultilinearExtension<S>& self) {
    return self;
  }
};

}  // namespace internal
}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_
