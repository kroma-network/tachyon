// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_
#define VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/zk/r1cs/constraint_system/quadratic_arithmetic_program.h"

namespace tachyon::circom {

template <typename F>
class QuadraticArithmeticProgram {
 public:
  QuadraticArithmeticProgram() = delete;

  template <typename Domain>
  static zk::r1cs::QAPInstanceMapResult<F> InstanceMap(
      const Domain* domain, const zk::r1cs::ConstraintSystem<F>& cs,
      const F& x) {
    return zk::r1cs::QuadraticArithmeticProgram<F>::InstanceMap(domain, cs, x);
  }

  template <typename Domain>
  static std::vector<F> WitnessMapFromMatrices(
      const Domain* domain, const zk::r1cs::ConstraintMatrices<F>& matrices,
      absl::Span<const F> full_assignments) {
    using Evals = typename Domain::Evals;
    using DensePoly = typename Domain::DensePoly;

    CHECK_GE(domain->size(), matrices.num_constraints);

    std::vector<F> a(domain->size());
    std::vector<F> b(domain->size());
    std::vector<F> c(domain->size());

    // clang-format off
    // |a[i]| = Σⱼ₌₀..ₘ (xⱼ * Aᵢ,ⱼ)    (if i < |num_constraints|)
    //        = x[i - num_constraints] (otherwise)
    // |b[i]| = Σⱼ₌₀..ₘ (xⱼ * Bᵢ,ⱼ)    (if i < |num_constraints|)
    //        = 0                      (otherwise)
    // |c[i]| = |a[i]|* |b[i]|         (if i < |num_constraints|)
    //        = 0                      (otherwise)
    // where x is |full_assignments|.
    // clang-format on
    OPENMP_PARALLEL_FOR(size_t i = 0; i < matrices.num_constraints; ++i) {
      a[i] = zk::r1cs::EvaluateConstraint(matrices.a[i], full_assignments);
      b[i] = zk::r1cs::EvaluateConstraint(matrices.b[i], full_assignments);
      c[i] = a[i] * b[i];
    }

    for (size_t i = matrices.num_constraints;
         i < matrices.num_constraints + matrices.num_instance_variables; ++i) {
      a[i] = full_assignments[i - matrices.num_constraints];
    }

    Evals a_evals(std::move(a));
    DensePoly a_poly = domain->IFFT(std::move(a_evals));
    Evals b_evals(std::move(b));
    DensePoly b_poly = domain->IFFT(std::move(b_evals));
    Evals c_evals(std::move(c));
    DensePoly c_poly = domain->IFFT(std::move(c_evals));

    F root_of_unity;
    {
      std::unique_ptr<Domain> extended_domain =
          Domain::Create(2 * domain->size());
      root_of_unity = extended_domain->GetElement(1);
    }
    Domain::DistributePowers(a_poly, root_of_unity);
    Domain::DistributePowers(b_poly, root_of_unity);
    Domain::DistributePowers(c_poly, root_of_unity);

    a_evals = domain->FFT(std::move(a_poly));
    b_evals = domain->FFT(std::move(b_poly));
    c_evals = domain->FFT(std::move(c_poly));

    // |h_evals[i]| = |a[i]| * |b[i]| - |c[i]|
    OPENMP_PARALLEL_FOR(size_t i = 0; i < domain->size(); ++i) {
      F& h_evals_i = a_evals.at(i);
      h_evals_i *= b_evals[i];
      h_evals_i -= c_evals[i];
    }

    return std::move(a_evals).TakeEvaluations();
  }

  template <typename Domain>
  static std::vector<F> ComputeHQuery(const Domain* domain, const F& t_x,
                                      const F& x, const F& delta_inverse) {
    using Evals = typename Domain::Evals;
    using DensePoly = typename Domain::DensePoly;

    // The usual H query has domain - 1 powers. Z has domain powers. So HZ has
    // 2 * domain - 1 powers.
    std::unique_ptr<Domain> extended_domain =
        Domain::Create(domain->size() * 2 + 1);
    Evals evals(
        F::GetSuccessivePowers(extended_domain->size(), t_x, delta_inverse));
    DensePoly poly = extended_domain->IFFT(std::move(evals));
    std::vector<F> ret(domain->size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < domain->size(); ++i) {
      ret[i] = poly[2 * i + 1];
    }
    return ret;
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_
