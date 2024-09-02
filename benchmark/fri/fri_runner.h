#ifndef BENCHMARK_FRI_FRI_RUNNER_H_
#define BENCHMARK_FRI_FRI_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/fri/fri_config.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/crypto/commitments/fri/two_adic_multiplicative_coset.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::benchmark {

template <typename PCS>
class FRIRunner {
 public:
  using F = typename PCS::F;
  using Domain = crypto::TwoAdicMultiplicativeCoset<F>;

  typedef tachyon_baby_bear* (*ExternalFn)(const tachyon_baby_bear* data,
                                           const size_t* degrees,
                                           size_t num_of_degrees,
                                           size_t batch_size,
                                           uint32_t log_blowup,
                                           uint64_t* duration);

  FRIRunner(SimpleReporter& reporter, const FRIConfig& config, PCS& pcs)
      : reporter_(reporter), config_(config), pcs_(pcs) {}

  F Run(Vendor vendor, const math::RowMajorMatrix<F>& input) {
    size_t max_degree = static_cast<size_t>(input.rows());
    std::vector<size_t> degrees = GetInputDegrees(max_degree);

    std::vector<math::RowMajorMatrix<F>> inner_polys =
        base::Map(degrees, [this, input](size_t degree) {
          math::RowMajorMatrix<F> ret =
              Eigen::Map<const math::RowMajorMatrix<F>>(
                  &input.data()[0], degree, config_.batch_size());
          return ret;
        });

    std::vector<Domain> inner_domains =
        base::Map(degrees, [this](size_t degree) {
          return this->pcs_.GetNaturalDomainForDegree(degree);
        });

    base::TimeTicks start = base::TimeTicks::Now();
    typename PCS::Commitment commit;
    typename PCS::ProverData prover_data;
    CHECK(pcs_.Commit(inner_domains, inner_polys, &commit, &prover_data));
    reporter_.AddTime(vendor, base::TimeTicks::Now() - start);
    return commit[1];
  }

  F RunExternal(Vendor vendor, ExternalFn fn,
                const math::RowMajorMatrix<F>& input) {
    size_t max_degree = static_cast<size_t>(input.rows());
    std::vector<size_t> degrees = GetInputDegrees(max_degree);

    uint64_t duration_in_us = 0;
    tachyon_baby_bear* data =
        fn(c::base::c_cast(input.data()), &degrees[0], degrees.size(),
           config_.batch_size(), config_.log_blowup(), &duration_in_us);
    reporter_.AddTime(vendor, base::Microseconds(duration_in_us));
    return c::base::native_cast(data)[1];
  }

 private:
  std::vector<size_t> GetInputDegrees(size_t max_degree) {
    std::vector<size_t> degrees;
    degrees.reserve(config_.input_num());
    for (size_t d = max_degree >> (config_.input_num() - 1); d <= max_degree;
         d <<= 1) {
      degrees.push_back(d);
    }
    return degrees;
  }

  SimpleReporter& reporter_;
  const FRIConfig& config_;
  PCS& pcs_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FRI_FRI_RUNNER_H_
