#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_LINEAR_COMBINATION_TERM_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_LINEAR_COMBINATION_TERM_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/parallelize.h"

namespace tachyon {
namespace math {

template <typename F>
struct LinearCombinationTerm {
  // Note(ashjeong): Member |indexes| relies on |LinearCombination|'s
  // |flattened_ml_evaluations|
  F coefficient;
  std::vector<size_t> indexes;

  template <typename Container, typename MLE>
  F Evaluate(
      const Container& point,
      const std::vector<std::shared_ptr<MLE>>& flattened_ml_evaluations) const {
    std::vector<F> results = base::ParallelizeMap(
        indexes,
        [&flattened_ml_evaluations, &point](absl::Span<const size_t> chunk) {
          return EvaluateSerial(point, flattened_ml_evaluations, chunk);
        });
    return coefficient * std::accumulate(results.begin(), results.end(),
                                         F::One(), std::multiplies<>());
  }

  template <typename MLE>
  F Combine(
      size_t num_variables_,
      const std::vector<std::shared_ptr<MLE>>& flattened_ml_evaluations) const {
    size_t parallel_factor = 16;
    CHECK(!indexes.empty());

#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
    size_t thread_nums = 1;
#endif
    size_t size = size_t{1} << num_variables_;
    thread_nums = ((thread_nums * parallel_factor) <= size) ? thread_nums : 1;

    size_t chunk_size = (size + thread_nums - 1) / thread_nums;
    size_t num_chunks = (size + chunk_size - 1) / chunk_size;

    std::vector<F> sums(num_chunks, F::Zero());
    OMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
      size_t start = i * chunk_size;
      size_t len = (i == num_chunks - 1) ? size - start : chunk_size;
      for (size_t j = start; j < start + len; ++j) {
        sums[i] += std::accumulate(
            indexes.begin(), indexes.end(), F::One(),
            [&flattened_ml_evaluations, j](F& acc, size_t index) {
              const std::vector<F>& evals =
                  flattened_ml_evaluations[index]->evaluations();
              return (j < evals.size()) ? (acc *= evals[j]) : acc;
            });
      }
    }
    F sum = std::accumulate(sums.begin(), sums.end(), F::Zero());
    return sum *= coefficient;
  }

  bool operator==(const LinearCombinationTerm& other) const {
    return coefficient == other.coefficient && indexes == other.indexes;
  }
  bool operator!=(const LinearCombinationTerm& other) const {
    return !operator==(other);
  }

 private:
  template <typename Container, typename MLE>
  static F EvaluateSerial(
      const Container& point,
      const std::vector<std::shared_ptr<MLE>>& flattened_ml_evaluations,
      absl::Span<const size_t> indexes) {
    return std::accumulate(
        indexes.begin(), indexes.end(), F::One(),
        [&flattened_ml_evaluations, &point](F& acc, const size_t index) {
          return acc *= flattened_ml_evaluations[index]->Evaluate(point);
        });
  }
};

}  // namespace math

namespace base {

template <typename F>
class Copyable<math::LinearCombinationTerm<F>> {
 public:
  static bool WriteTo(const math::LinearCombinationTerm<F>& term,
                      Buffer* buffer) {
    return buffer->WriteMany(term.coefficient, term.indexes);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       math::LinearCombinationTerm<F>* term) {
    F coefficient;
    std::vector<size_t> indexes;
    if (!buffer.ReadMany(&coefficient, &indexes)) return false;
    *term = {std::move(coefficient), std::move(indexes)};
    return true;
  }

  static size_t EstimateSize(const math::LinearCombinationTerm<F>& term) {
    return base::EstimateSize(term.coefficient, term.indexes);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_LINEAR_COMBINATION_TERM_H_
