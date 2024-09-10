#ifndef BENCHMARK_FFT_BATCH_FFT_BATCH_RUNNER_H_
#define BENCHMARK_FFT_BATCH_FFT_BATCH_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/fft_batch/fft_batch_config.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/matrix/baby_bear_row_major_matrix_type_traits.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/polynomials/univariate/radix2_evaluation_domain.h"

namespace tachyon::benchmark {

template <typename Domain>
class FFTBatchRunner {
 public:
  using F = typename Domain::Field;

  typedef tachyon_baby_bear* (*ExternalFn)(const tachyon_baby_bear* data,
                                           uint32_t n_log, size_t batch_size,
                                           uint64_t* duration);
  FFTBatchRunner(SimpleReporter& reporter, const FFTBatchConfig& config)
      : reporter_(reporter), config_(config) {}

  math::RowMajorMatrix<F> Run(Vendor vendor, bool run_coset_lde,
                              const math::RowMajorMatrix<F>& input) {
    math::RowMajorMatrix<F> result;
    std::unique_ptr<Domain> domain =
        Domain::Create(static_cast<size_t>(input.rows()));
    base::TimeTicks start = base::TimeTicks::Now();
    if (run_coset_lde) {
      const size_t kAddedBits = 1;
      math::RowMajorMatrix<F> input_tmp = input;
      result =
          math::RowMajorMatrix<F>(input.rows() << kAddedBits, input.cols());
      domain->CosetLDEBatch(input_tmp, kAddedBits,
                            F::FromMontgomery(F::Config::kSubgroupGenerator),
                            result);
    } else {
      result = input;
      domain->FFTBatch(result);
    }
    reporter_.AddTime(vendor, base::TimeTicks::Now() - start);
    return result;
  }

  math::RowMajorMatrix<F> RunExternal(Vendor vendor, ExternalFn fn,
                                      const math::RowMajorMatrix<F>& input) {
    uint64_t duration_in_us;
    std::unique_ptr<tachyon_baby_bear_row_major_matrix> ret;
    tachyon_baby_bear* data = fn(c::base::c_cast(input.data()), input.rows(),
                                 input.cols(), &duration_in_us);
    ret.reset(tachyon_baby_bear_row_major_matrix_create(data, input.rows(),
                                                        input.cols()));
    reporter_.AddTime(vendor, base::Microseconds(duration_in_us));
    return std::move(*c::base::native_cast(ret.get()));
  }

 private:
  SimpleReporter& reporter_;
  const FFTBatchConfig& config_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FFT_BATCH_FFT_BATCH_RUNNER_H_
