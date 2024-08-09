#ifndef BENCHMARK_FFT_BATCH_BENCHMARK_RUNNER_H_
#define BENCHMARK_FFT_BATCH_BENCHMARK_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "benchmark/fft_batch/benchmark_reporter.h"
#include "benchmark/fft_batch/config.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/polynomials/univariate/radix2_evaluation_domain.h"

namespace tachyon::benchmark::fft_batch {

template <typename F>
class BenchmarkRunner {
 public:
  // TODO(batzor): implement |CRowMajorMatrix| and use it to check results
  typedef void* (*ExternalFn)(uint64_t* duration, F* data, size_t n_log,
                              size_t batch_size);
  BenchmarkRunner(SimpleBenchmarkReporter* reporter, const Config* config)
      : reporter_(reporter), config_(config) {}

  void Prepare() {
    size_t n = size_t{1} << config_->degree();
    input_ = math::RowMajorMatrix<F>::Random(n, config_->batch_size());
    domain_ = math::Radix2EvaluationDomain<F>::Create(n);
  }

  math::RowMajorMatrix<F> Run() {
    CHECK(domain_) << "Prepare() must be called before Run()";
    math::RowMajorMatrix<F> result;
    for (size_t i = 0; i < config_->repeating_num(); ++i) {
      math::RowMajorMatrix<F> matrix = input_;
      base::TimeTicks start = base::TimeTicks::Now();
      domain_->FFTBatch(input_);
      reporter_->AddTime(i, (base::TimeTicks::Now() - start).InSecondsF());
      if (i == 0) {
        result = matrix;
      }
    }
    return result;
  }

  math::RowMajorMatrix<F> RunExternal(ExternalFn fn) {
    for (size_t i = 0; i < config_->repeating_num(); ++i) {
      uint64_t duration_in_us;
      fn(&duration_in_us, input_.data(), config_->degree(),
         config_->batch_size());
      reporter_->AddTime(i, base::Microseconds(duration_in_us).InSecondsF());
    }
    // TODO(batzor): use result from |fn| and convert it to |RowMajorMatrix|
    return math::RowMajorMatrix<F>();
  }

 private:
  // not owned
  SimpleBenchmarkReporter* const reporter_;
  // not owned
  const Config* const config_;
  math::RowMajorMatrix<F> input_;
  std::unique_ptr<math::Radix2EvaluationDomain<F>> domain_;
};

}  // namespace tachyon::benchmark::fft_batch

#endif  // BENCHMARK_FFT_BATCH_BENCHMARK_RUNNER_H_
