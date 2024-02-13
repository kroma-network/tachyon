#ifndef BENCHMARK_FFT_FFT_RUNNER_H_
#define BENCHMARK_FFT_FFT_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

// clang-format off
#include "benchmark/fft/simple_fft_benchmark_reporter.h"
// clang-format on
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon {

template <typename F>
class FFTRunner {
 public:
  constexpr static size_t kMaxDegree = SIZE_MAX - 1;
  using Domain = math::UnivariateEvaluationDomain<F, kMaxDegree>;
  using DensePoly = typename Domain::DensePoly;

  typedef tachyon_bn254_fr* (*FFTExternalFn)(const tachyon_bn254_fr* coeffs,
                                             size_t n,
                                             const tachyon_bn254_fr* omega,
                                             uint32_t k,
                                             uint64_t* duration_in_us);

  explicit FFTRunner(SimpleFFTBenchmarkReporter* reporter)
      : reporter_(reporter) {}

  void SetInput(const DensePoly* poly, std::unique_ptr<Domain> domain) {
    poly_ = poly;
    domain_ = std::move(domain);
  }

  template <typename Fn>
  void Run(Fn fn) {
    base::TimeTicks now = base::TimeTicks::Now();
    fn(reinterpret_cast<const tachyon_bn254_univariate_evaluation_domain*>(
           domain_.get()),
       reinterpret_cast<const tachyon_bn254_univariate_dense_polynomial*>(
           poly_));
    reporter_->AddResult(0, (base::TimeTicks::Now() - now).InSecondsF());
  }

  void RunExternal(FFTExternalFn fn) const {
    base::TimeTicks now = base::TimeTicks::Now();
    uint64_t duration_in_us;
    const F omega = domain_->group_gen();
    fn(reinterpret_cast<const tachyon_bn254_fr*>(
           poly_->coefficients().coefficients().data()),
       poly_->Degree(), reinterpret_cast<const tachyon_bn254_fr*>(&omega),
       domain_->log_size_of_group(), &duration_in_us);
    reporter_->AddResult(0, (base::TimeTicks::Now() - now).InSecondsF());
  }

 private:
  // not owned
  SimpleFFTBenchmarkReporter* reporter_ = nullptr;
  // not owned
  const DensePoly* poly_ = nullptr;
  // not owned
  std::unique_ptr<Domain> domain_;
};

}  // namespace tachyon

#endif  // BENCHMARK_fft_FFT_RUNNER_H_
