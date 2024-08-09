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
    constexpr size_t kParallelFactor = 16;
    CHECK(!indexes.empty());

#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
    size_t thread_nums = 1;
#endif

    size_t size = size_t{1} << num_variables_;
    std::vector<F> sums = base::ParallelizeMap(
        size,
        [this, &flattened_ml_evaluations](size_t len, size_t chunk_offset,
                                          size_t chunk_size) {
          size_t start = chunk_offset * chunk_size;
          F sum = F::Zero();
          for (size_t i = start; i < start + len; ++i) {
            sum += std::accumulate(
                indexes.begin(), indexes.end(), F::One(),
                [&flattened_ml_evaluations, i](F& acc, size_t index) {
                  const std::vector<F>& evals =
                      flattened_ml_evaluations[index]->evaluations();
                  return (i < evals.size()) ? (acc *= evals[i]) : acc;
                });
          }
          return sum;
        },
        kParallelFactor * thread_nums);
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
