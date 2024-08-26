#ifndef BENCHMARK_FFT_FFT_RUNNER_H_
#define BENCHMARK_FFT_FFT_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/functional/functor_traits.h"
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"

#if TACHYON_CUDA
#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt_holder.h"
#endif

namespace tachyon::benchmark {

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

  explicit FFTRunner(SimpleReporter& reporter) : reporter_(reporter) {}

  template <typename Fn, typename RetPoly,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Fn>,
            typename RunType = typename FunctorTraits::RunType,
            typename ArgList = base::internal::ExtractArgs<RunType>,
            typename CRetPoly = std::conditional_t<
                std::is_same_v<RetPoly, typename Domain::Evals>,
                tachyon_bn254_univariate_evaluations,
                tachyon_bn254_univariate_dense_polynomial>>
  void Run(Vendor vendor, Fn fn, Domain* domain, const PolyOrEvals& input,
           bool should_record, RetPoly& result) {
    PolyOrEvals poly = input;

    base::TimeTicks now = base::TimeTicks::Now();
    std::unique_ptr<CRetPoly> ret;
    ret.reset(fn(c::base::c_cast(domain), c::base::c_cast(&poly)));
    if (should_record) {
      reporter_.AddTime(vendor, (base::TimeTicks::Now() - now));
    }

    result = std::move(*c::base::native_cast(ret.get()));
  }

  template <typename RetPoly>
  void RunExternal(Vendor vendor, FFTExternalFn fn, Domain* domain,
                   const PolyOrEvals& input, RetPoly& result) const {
    uint64_t duration_in_us;

    std::unique_ptr<F> ret;
    if constexpr (std::is_same_v<PolyOrEvals, typename Domain::Evals>) {
      const F& omega_inv = domain->group_gen_inv();
      ret.reset(c::base::native_cast(
          fn(c::base::c_cast(input.evaluations().data()), input.NumElements(),
             c::base::c_cast(&omega_inv), domain->log_size_of_group(),
             &duration_in_us)));
      std::vector<F> res_vec(ret.get(), ret.get() + input.NumElements());
      reporter_.AddTime(vendor, base::Microseconds(duration_in_us));
      result = RetPoly(typename RetPoly::Coefficients(std::move(res_vec)));
      // NOLINTNEXTLINE(readability/braces)
    } else if constexpr (std::is_same_v<PolyOrEvals,
                                        typename Domain::DensePoly>) {
      const F& omega = domain->group_gen();
      ret.reset(c::base::native_cast(
          fn(c::base::c_cast(input.coefficients().coefficients().data()),
             input.NumElements(), c::base::c_cast(&omega),
             domain->log_size_of_group(), &duration_in_us)));
      std::vector<F> res_vec(ret.get(), ret.get() + input.NumElements());
      reporter_.AddTime(vendor, base::Microseconds(duration_in_us));
      result = RetPoly(std::move(res_vec));
    }
  }

 private:
  SimpleReporter& reporter_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FFT_FFT_RUNNER_H_
