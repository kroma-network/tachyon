#ifndef BENCHMARK_MSM_MSM_RUNNER_H_
#define BENCHMARK_MSM_MSM_RUNNER_H_

#include <stddef.h>

#include <memory>
#include <vector>

// clang-format off
#include "benchmark/msm/simple_msm_benchmark_reporter.h"
// clang-format on
#include "tachyon/base/time/time.h"
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/math/base/semigroups.h"

namespace tachyon {

template <typename Point>
class MSMRunner {
 public:
  using ScalarField = typename Point::ScalarField;
  using RetPoint =
      typename math::internal::AdditiveSemigroupTraits<Point>::ReturnTy;

  using CPoint = typename c::math::PointTraits<Point>::CCurvePoint;
  using CRetPoint = typename c::math::PointTraits<RetPoint>::CCurvePoint;
  using CScalarField = typename c::math::PointTraits<Point>::CScalarField;

  typedef CRetPoint* (*MSMAffineExternalFn)(const CPoint* bases,
                                            const CScalarField* scalars,
                                            size_t size,
                                            uint64_t* duration_in_us);

  explicit MSMRunner(SimpleMSMBenchmarkReporter* reporter)
      : reporter_(reporter) {}

  void SetInputs(const std::vector<Point>* bases,
                 const std::vector<ScalarField>* scalars) {
    bases_ = bases;
    scalars_ = scalars;
  }

  template <typename Fn, typename MSMPtr>
  void Run(Fn fn, MSMPtr msm, const std::vector<uint64_t>& point_nums,
           std::vector<RetPoint>* results) {
    results->clear();
    for (size_t i = 0; i < point_nums.size(); ++i) {
      base::TimeTicks now = base::TimeTicks::Now();
      std::unique_ptr<CRetPoint> ret;
      ret.reset(fn(msm, c::base::c_cast(bases_->data()),
                   c::base::c_cast(scalars_->data()), point_nums[i]));
      reporter_->AddTime(i, (base::TimeTicks::Now() - now).InSecondsF());
      results->push_back(*c::base::native_cast(ret.get()));
    }
  }

  void RunExternal(MSMAffineExternalFn fn,
                   const std::vector<uint64_t>& point_nums,
                   std::vector<RetPoint>* results) const {
    for (size_t i = 0; i < point_nums.size(); ++i) {
      std::unique_ptr<CRetPoint> ret;
      uint64_t duration_in_us;
      ret.reset(fn(c::base::c_cast(bases_->data()),
                   c::base::c_cast(scalars_->data()), point_nums[i],
                   &duration_in_us));
      reporter_->AddTime(i, base::Microseconds(duration_in_us).InSecondsF());
      results->push_back(*c::base::native_cast(ret.get()));
    }
  }

 private:
  // not owned
  SimpleMSMBenchmarkReporter* const reporter_;
  // not owned
  const std::vector<Point>* bases_ = nullptr;
  // not owned
  const std::vector<ScalarField>* scalars_ = nullptr;
};

}  // namespace tachyon

#endif  // BENCHMARK_MSM_MSM_RUNNER_H_
