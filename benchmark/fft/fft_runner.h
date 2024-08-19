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

  void set_polys(absl::Span<const PolyOrEvals> polys) { polys_ = polys; }

  void set_domains(absl::Span<std::unique_ptr<Domain>> domains) {
    domains_ = domains;
  }

#if TACHYON_CUDA
  void SwitchToIcicle(math::IcicleNTTHolder<F>* icicle_ntt_holder) {
    for (std::unique_ptr<Domain>& domain : domains_) {
      domain->set_icicle(icicle_ntt_holder);
    }
  }
#endif

  template <typename Fn, typename RetPoly,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Fn>,
            typename RunType = typename FunctorTraits::RunType,
            typename ArgList = base::internal::ExtractArgs<RunType>,
            typename CPolyOrEvals = base::internal::GetType<1, ArgList>,
            typename CRetPoly = std::conditional_t<
                std::is_same_v<RetPoly, typename Domain::Evals>,
                tachyon_bn254_univariate_evaluations,
                tachyon_bn254_univariate_dense_polynomial>>
  void Run(Vendor vendor, Fn fn, const std::vector<size_t>& degrees,
           std::vector<RetPoly>& results, bool should_record) {
    if (should_record) {
      reporter_.AddVendor(vendor);
    }

    results.clear();
    results.reserve(degrees.size());
    for (size_t i = 0; i < degrees.size(); ++i) {
      PolyOrEvals poly = polys_[i];
      base::TimeTicks now = base::TimeTicks::Now();
      std::unique_ptr<CRetPoly> ret;
      ret.reset(fn(c::base::c_cast(domains_[i].get()), c::base::c_cast(&poly)));
      if (should_record) {
        reporter_.AddTime(vendor, (base::TimeTicks::Now() - now));
      }
      results.push_back(*c::base::native_cast(ret.get()));
    }
  }

  template <typename RetPoly>
  void RunExternal(Vendor vendor, FFTExternalFn fn,
                   const std::vector<size_t>& exponents,
                   std::vector<RetPoly>& results) const {
    reporter_.AddVendor(vendor);

    results.clear();
    results.reserve(exponents.size());
    for (size_t i = 0; i < exponents.size(); ++i) {
      uint64_t duration_in_us;
      size_t n = size_t{1} << exponents[i];

      std::unique_ptr<F> ret;
      if constexpr (std::is_same_v<PolyOrEvals, typename Domain::Evals>) {
        const F& omega_inv = domains_[i]->group_gen_inv();
        ret.reset(c::base::native_cast(
            fn(c::base::c_cast(polys_[i].evaluations().data()), n,
               c::base::c_cast(&omega_inv), exponents[i], &duration_in_us)));
        std::vector<F> res_vec(ret.get(), ret.get() + n);
        results.emplace_back(
            typename RetPoly::Coefficients(std::move(res_vec)));
        // NOLINTNEXTLINE(readability/braces)
      } else if constexpr (std::is_same_v<PolyOrEvals,
                                          typename Domain::DensePoly>) {
        const F& omega = domains_[i]->group_gen();
        ret.reset(c::base::native_cast(
            fn(c::base::c_cast(polys_[i].coefficients().coefficients().data()),
               n, c::base::c_cast(&omega), exponents[i], &duration_in_us)));
        std::vector<F> res_vec(ret.get(), ret.get() + n);
        results.emplace_back(std::move(res_vec));
      }
      reporter_.AddTime(vendor, base::Microseconds(duration_in_us));
    }
  }

 private:
  SimpleReporter& reporter_;
  absl::Span<const PolyOrEvals> polys_;
  absl::Span<std::unique_ptr<Domain>> domains_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FFT_FFT_RUNNER_H_
