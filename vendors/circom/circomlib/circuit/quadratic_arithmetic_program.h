// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_
#define VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_

#include <memory_resource>
#include <utility>
#include <vector>

#include "circomlib/zkey/coefficient.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/r1cs/constraint_system/quadratic_arithmetic_program.h"

namespace tachyon::circom {

template <typename F>
class QuadraticArithmeticProgram {
 public:
  QuadraticArithmeticProgram() = delete;

  template <typename Domain>
  static std::pmr::vector<F> WitnessMapFromMatrices(
      const Domain* domain, absl::Span<const Coefficient<F>> coefficients,
      absl::Span<const F> full_assignments) {
    using Evals = typename Domain::Evals;
    using DensePoly = typename Domain::DensePoly;

    std::vector<Evals> abc;
    abc.reserve(3);
    abc.emplace_back(std::pmr::vector<F>(domain->size()));
    abc.emplace_back(std::pmr::vector<F>(domain->size()));
    abc.emplace_back(std::pmr::vector<F>(domain->size()));

    // See
    // https://github.com/iden3/rapidsnark/blob/b17e6fe/src/groth16.cpp#L116-L156.
#if defined(TACHYON_HAS_OPENMP)
    constexpr size_t kNumLocks = 1024;
    omp_lock_t locks[kNumLocks];
    for (size_t i = 0; i < kNumLocks; i++) omp_init_lock(&locks[i]);
#endif
    OPENMP_PARALLEL_FOR(size_t i = 0; i < coefficients.size(); i++) {
      const Coefficient<F>& c = coefficients[i];
      std::pmr::vector<F>& ab =
          (c.matrix == 0) ? abc[0].evaluations() : abc[1].evaluations();

#if defined(TACHYON_HAS_OPENMP)
      omp_set_lock(&locks[c.constraint % kNumLocks]);
#endif
      if (c.value.IsOne()) {
        ab[c.constraint] += full_assignments[c.signal];
      } else {
        ab[c.constraint] += c.value * full_assignments[c.signal];
      }
#if defined(TACHYON_HAS_OPENMP)
      omp_unset_lock(&locks[c.constraint % kNumLocks]);
#endif
    }
#if defined(TACHYON_HAS_OPENMP)
    for (size_t i = 0; i < kNumLocks; i++) omp_destroy_lock(&locks[i]);
#endif

    OPENMP_PARALLEL_FOR(size_t i = 0; i < domain->size(); ++i) {
      abc[2].evaluations()[i] =
          abc[0].evaluations()[i] * abc[1].evaluations()[i];
    }

    std::vector<DensePoly> abc_polys = domain->IFFT(std::move(abc));

    F root_of_unity;
    CHECK(F::GetRootOfUnity(2 * domain->size(), &root_of_unity));

    Domain::DistributePowers(abc_polys, root_of_unity);

    std::vector<Evals> abc_evals = domain->FFT(std::move(abc_polys));

    // |h_evals[i]| = |a[i]| * |b[i]| - |c[i]|
    OPENMP_PARALLEL_FOR(size_t i = 0; i < domain->size(); ++i) {
      F& h_evals_i = abc_evals[0].at(i);
      h_evals_i *= abc_evals[1][i];
      h_evals_i -= abc_evals[2][i];
    }

    return std::move(abc_evals[0]).TakeEvaluations();
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_
