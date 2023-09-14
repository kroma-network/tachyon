#ifndef BENCHMARK_MSM_MSM_RUNNER_H_
#define BENCHMARK_MSM_MSM_RUNNER_H_

#include <stddef.h>

#include <vector>

// clang-format off
#include "benchmark/msm/simple_msm_benchmark_reporter.h"
// clang-format on
#include "tachyon/base/time/time.h"
#include "tachyon/cc/math/elliptic_curves/point_traits.h"
#include "tachyon/math/base/semigroups.h"

namespace tachyon {

template <typename PointTy>
class MSMRunner {
 public:
  using ScalarField = typename PointTy::ScalarField;
  using ReturnTy =
      typename math::internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;

  using CPointTy = typename cc::math::PointTraits<PointTy>::CCurvePointTy;
  using CReturnTy = typename cc::math::PointTraits<ReturnTy>::CCurvePointTy;
  using CScalarField = typename cc::math::PointTraits<PointTy>::CScalarField;

  typedef CReturnTy* (*MSMAffineExternalFn)(const CPointTy* bases,
                                            size_t bases_len,
                                            const CScalarField* scalars,
                                            size_t scalars_len,
                                            uint64_t* duration_in_us);

  typedef CReturnTy* (*MSMAffineFn)(const CPointTy* bases, size_t bases_len,
                                    const CScalarField* scalars,
                                    size_t scalars_len);

  explicit MSMRunner(SimpleMSMBenchmarkReporter* reporter)
      : reporter_(reporter) {}

  void SetInputs(const std::vector<PointTy>* bases,
                 const std::vector<ScalarField>* scalars) {
    bases_ = bases;
    scalars_ = scalars;
  }

  void Run(MSMAffineFn fn, const std::vector<uint64_t>& point_nums,
           std::vector<ReturnTy>* results) {
    results->clear();
    for (size_t i = 0; i < point_nums.size(); ++i) {
      base::TimeTicks now = base::TimeTicks::Now();
      std::unique_ptr<CReturnTy> ret;
      ret.reset(fn(reinterpret_cast<const CPointTy*>(bases_->data()),
                   point_nums[i],
                   reinterpret_cast<const CScalarField*>(scalars_->data()),
                   point_nums[i]));
      reporter_->AddResult(i, (base::TimeTicks::Now() - now).InSecondsF());
      results->push_back(*reinterpret_cast<ReturnTy*>(ret.get()));
    }
  }

  void RunExternal(MSMAffineExternalFn fn,
                   const std::vector<uint64_t>& point_nums,
                   std::vector<ReturnTy>* results) const {
    for (size_t i = 0; i < point_nums.size(); ++i) {
      std::unique_ptr<CReturnTy> ret;
      uint64_t duration_in_us;
      ret.reset(fn(reinterpret_cast<const CPointTy*>(bases_->data()),
                   point_nums[i],
                   reinterpret_cast<const CScalarField*>(scalars_->data()),
                   point_nums[i], &duration_in_us));
      results->push_back(*reinterpret_cast<ReturnTy*>(ret.get()));
      reporter_->AddResult(i, base::Microseconds(duration_in_us).InSecondsF());
    }
  }

 private:
  // not owned
  SimpleMSMBenchmarkReporter* reporter_ = nullptr;
  // not owned
  const std::vector<PointTy>* bases_ = nullptr;
  // not owned
  const std::vector<ScalarField>* scalars_ = nullptr;
};

}  // namespace tachyon

#endif  // BENCHMARK_MSM_MSM_RUNNER_H_
