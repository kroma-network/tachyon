#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_PIPPENGER_ADAPTER_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_PIPPENGER_ADAPTER_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger.h"

namespace tachyon::math {

enum class PippengerParallelStrategy {
  kNone,
  kParallelWindow,
  kParallelTerm,
  kParallelWindowAndTerm,
};

template <typename Point>
class PippengerAdapter {
 public:
  using ScalarField = typename Point::ScalarField;
  using Bucket = typename Pippenger<Point>::Bucket;

  template <typename BaseInputIterator, typename ScalarInputIterator>
  [[nodiscard]] bool Run(BaseInputIterator bases_first,
                         BaseInputIterator bases_last,
                         ScalarInputIterator scalars_first,
                         ScalarInputIterator scalars_last, Bucket* ret) {
    return RunWithStrategy(std::move(bases_first), std::move(bases_last),
                           std::move(scalars_first), std::move(scalars_last),
                           PippengerParallelStrategy::kParallelTerm, ret);
  }

  template <typename BaseInputIterator, typename ScalarInputIterator>
  [[nodiscard]] bool RunWithStrategy(BaseInputIterator bases_first,
                                     BaseInputIterator bases_last,
                                     ScalarInputIterator scalars_first,
                                     ScalarInputIterator scalars_last,
                                     PippengerParallelStrategy strategy,
                                     Bucket* ret) {
    if (strategy == PippengerParallelStrategy::kNone ||
        strategy == PippengerParallelStrategy::kParallelWindow) {
      Pippenger<Point> pippenger;
      pippenger.SetParallelWindows(strategy ==
                                   PippengerParallelStrategy::kParallelWindow);
      return pippenger.Run(std::move(bases_first), std::move(bases_last),
                           std::move(scalars_first), std::move(scalars_last),
                           ret);
    } else {
      size_t bases_size = std::distance(bases_first, bases_last);
      size_t scalars_size = std::distance(scalars_first, scalars_last);
      if (bases_size != scalars_size) {
        LOG(ERROR) << "bases_size and scalars_size don't match";
        return false;
      }
      if (scalars_size == 0) {
        *ret = Bucket::Zero();
        return true;
      }

#if defined(TACHYON_HAS_OPENMP)
      int thread_nums = omp_get_max_threads();
      if (strategy == PippengerParallelStrategy::kParallelWindowAndTerm) {
        size_t window_bits = MSMCtx::ComputeWindowsBits(scalars_size);
        size_t window_size =
            MSMCtx::ComputeWindowsCount<ScalarField>(window_bits);
        thread_nums = std::max(thread_nums / static_cast<int>(window_size), 2);
      }
#else
      int thread_nums = 1;
#endif  // defined(TACHYON_HAS_OPENMP)
      struct Result {
        Bucket value;
        bool valid;
      };

      std::vector<Result> results;
      results.resize(thread_nums);
      size_t size = (scalars_size + thread_nums - 1) / thread_nums;
#if defined(TACHYON_HAS_OPENMP)
      omp_set_num_threads(thread_nums);
#endif
      OPENMP_PARALLEL_FOR(int i = 0; i < thread_nums; ++i) {
        Pippenger<Point> pippenger;
        pippenger.SetParallelWindows(
            strategy == PippengerParallelStrategy::kParallelWindowAndTerm);
        auto bases_start = bases_first + size * i;
        auto bases_end =
            i == thread_nums - 1 ? bases_last : bases_first + size * (i + 1);
        auto scalars_start = scalars_first + size * i;
        auto scalars_end = i == thread_nums - 1
                               ? scalars_last
                               : scalars_first + size * (i + 1);
        results[i].valid = pippenger.Run(bases_start, bases_end, scalars_start,
                                         scalars_end, &results[i].value);
      }

      bool all_good =
          std::all_of(results.begin(), results.end(),
                      [](const Result& result) { return result.valid; });
      if (!all_good) return false;

      *ret = std::accumulate(results.begin(), results.end(), Bucket::Zero(),
                             [](Bucket& total, const Result& result) {
                               return total += result.value;
                             });
      return true;
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_PIPPENGER_ADAPTER_H_
