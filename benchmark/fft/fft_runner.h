#ifndef BENCHMARK_FFT_FFT_RUNNER_H_
#define BENCHMARK_FFT_FFT_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/fft/simple_fft_benchmark_reporter.h"
// clang-format on
#include "tachyon/base/functional/functor_traits.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"

namespace tachyon {

// NOTE(TomTaehoonKim): |PolyOrEvals| is the type of the input polynomial
// |polys_|. It can be either |Evals|, which is in the evaluation form, or
// |DensePoly|, which is in the coefficient form. |FFTRunner| is implemented
// this way to avoid duplication of the code for benchmarking FFT and IFFT.
template <typename Domain, typename PolyOrEvals>
class FFTRunner {
 public:
  using F = typename Domain::Field;
  typedef tachyon_bn254_fr* (*FFTExternalFn)(
      const tachyon_bn254_fr* coeffs, size_t n,
      const tachyon_bn254_fr* omega_or_omega_inv, uint32_t k,
      uint64_t* duration_in_us);

  explicit FFTRunner(SimpleFFTBenchmarkReporter* reporter)
      : reporter_(reporter) {}

  void SetInputs(const std::vector<PolyOrEvals>* polys,
                 std::vector<std::unique_ptr<Domain>>&& domains) {
    polys_ = polys;
    domains_ = std::move(domains);
  }

  template <typename Fn, typename RetPoly,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Fn>,
            typename RunType = typename FunctorTraits::RunType,
            typename ArgList = base::internal::ExtractArgs<RunType>,
            typename CPolyOrEvals = base::internal::GetType<1, ArgList>,
            typename CRetPoly = std::conditional_t<
                std::is_same_v<RetPoly, typename Domain::Evals>,
                tachyon_bn254_univariate_evaluations,
                tachyon_bn254_univariate_dense_polynomial>>
  void Run(Fn fn, const std::vector<uint64_t>& degrees,
           std::vector<RetPoly>* results) {
    for (size_t i = 0; i < degrees.size(); ++i) {
      base::TimeTicks now = base::TimeTicks::Now();
      std::unique_ptr<CRetPoly> ret;
      ret.reset(fn(
          reinterpret_cast<const tachyon_bn254_univariate_evaluation_domain*>(
              domains_[i].get()),
          reinterpret_cast<CPolyOrEvals>(&(*polys_)[i])));
      reporter_->AddResult(i, (base::TimeTicks::Now() - now).InSecondsF());
      results->push_back(*reinterpret_cast<RetPoly*>(ret.get()));
    }
  }

  template <typename RetPoly>
  void RunExternal(FFTExternalFn fn, const std::vector<uint64_t>& exponents,
                   std::vector<RetPoly>* results) const {
    for (size_t i = 0; i < exponents.size(); ++i) {
      uint64_t duration_in_us;
      const F omega = domains_[i]->group_gen();

      std::unique_ptr<F> ret;
      if constexpr (std::is_same_v<PolyOrEvals, typename Domain::Evals>) {
        const F omega_inv = omega.Inverse();
        ret.reset(reinterpret_cast<F*>(
            fn(reinterpret_cast<const tachyon_bn254_fr*>(
                   (*polys_)[i].evaluations().data()),
               (*polys_)[i].Degree(),
               reinterpret_cast<const tachyon_bn254_fr*>(&omega_inv),
               exponents[i], &duration_in_us)));
        std::vector<F> res_vec(ret.get(), ret.get() + (*polys_)[i].Degree());
        results->emplace_back(
            typename RetPoly::Coefficients(std::move(res_vec)));
        // NOLINTNEXTLINE(readability/braces)
      } else if constexpr (std::is_same_v<PolyOrEvals,
                                          typename Domain::DensePoly>) {
        ret.reset(reinterpret_cast<F*>(
            fn(reinterpret_cast<const tachyon_bn254_fr*>(
                   (*polys_)[i].coefficients().coefficients().data()),
               (*polys_)[i].Degree(),
               reinterpret_cast<const tachyon_bn254_fr*>(&omega), exponents[i],
               &duration_in_us)));
        std::vector<F> res_vec(ret.get(), ret.get() + (*polys_)[i].Degree());
        results->emplace_back(std::move(res_vec));
      }
      reporter_->AddResult(i, base::Microseconds(duration_in_us).InSecondsF());
    }
  }

 private:
  // not owned
  SimpleFFTBenchmarkReporter* reporter_ = nullptr;
  // not owned
  const std::vector<PolyOrEvals>* polys_ = nullptr;
  std::vector<std::unique_ptr<Domain>> domains_;
};

}  // namespace tachyon

#endif  // BENCHMARK_FFT_FFT_RUNNER_H_
