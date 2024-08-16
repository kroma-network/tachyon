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

  void set_inputs(absl::Span<math::RowMajorMatrix<F>> inputs) {
    inputs_ = inputs;
  }

  void set_domains(absl::Span<std::unique_ptr<Domain>> domains) {
    domains_ = domains;
  }

  void Run(Vendor vendor, std::vector<math::RowMajorMatrix<F>>& results,
           bool run_coset_lde) {
    results.clear();
    results.reserve(domains_.size());
    reporter_.AddVendor(vendor);
    for (size_t i = 0; i < domains_.size(); ++i) {
      math::RowMajorMatrix<F> matrix = inputs_[i];
      base::TimeTicks start = base::TimeTicks::Now();
      if (run_coset_lde) {
        domains_[i]->CosetLDEBatch(matrix, 0, F::Zero());
      } else {
        domains_[i]->FFTBatch(matrix);
      }
      reporter_.AddTime(vendor, base::TimeTicks::Now() - start);
      results.emplace_back(std::move(matrix));
    }
  }

  void RunExternal(Vendor vendor, ExternalFn fn,
                   std::vector<math::RowMajorMatrix<F>>& results) {
    results.clear();
    results.reserve(domains_.size());
    reporter_.AddVendor(vendor);
    for (size_t i = 0; i < domains_.size(); ++i) {
      uint64_t duration_in_us;
      std::unique_ptr<tachyon_baby_bear_row_major_matrix> ret;
      tachyon_baby_bear* data =
          fn(c::base::c_cast(inputs_[i].data()), config_.exponents()[i],
             config_.batch_size(), &duration_in_us);
      ret.reset(tachyon_baby_bear_row_major_matrix_create(
          data, inputs_[i].rows(), inputs_[i].cols()));
      reporter_.AddTime(vendor, base::Microseconds(duration_in_us));
      results.push_back(std::move(*c::base::native_cast(ret.get())));
    }
  }

 private:
  SimpleReporter& reporter_;
  const FFTBatchConfig& config_;

  absl::Span<const math::RowMajorMatrix<F>> inputs_;
  absl::Span<std::unique_ptr<Domain>> domains_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FFT_BATCH_FFT_BATCH_RUNNER_H_
