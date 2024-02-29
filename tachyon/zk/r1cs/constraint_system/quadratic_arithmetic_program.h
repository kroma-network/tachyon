// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_

#include <stddef.h>

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "tachyon/base/parallelize.h"
#include "tachyon/zk/r1cs/constraint_system/constraint_system.h"

namespace tachyon::zk::r1cs {

template <typename F>
class QuadraticArithmeticProgram {
 public:
  struct InstanceMapResult {
    std::vector<F> a;
    std::vector<F> b;
    std::vector<F> c;
    // t(x) = xⁿ⁺ˡ⁺¹ - 1
    F t_x;
    // n
    size_t num_constraints;
    // l + 1
    size_t num_instance_variables;
    // m - l
    size_t num_witness_variables;
    // m
    size_t num_qap_variables;
  };

  template <typename Poly>
  struct WitnessMapResult {
    Poly h;
    std::vector<F> full_assignments;
  };

  QuadraticArithmeticProgram() = delete;

  // Computes a QAP instance corresponding to the R1CS instance defined by |cs|.
  template <typename Domain>
  static InstanceMapResult InstanceMap(const Domain* domain,
                                       const ConstraintSystem<F>& cs,
                                       const F& x) {
    std::optional<ConstraintMatrices<F>> matrices = cs.ToMatrices();
    // |num_constraint| = n
    size_t num_constraints = cs.num_constraints();
    // |num_instance_variables| = l + 1
    size_t num_instance_variables = cs.num_instance_variables();
    // |num_witness_variables| = m - l
    size_t num_witness_variables = cs.num_witness_variables();

    // t(x) = xⁿ⁺ˡ⁺¹ - 1
    F t_x = domain->EvaluateVanishingPolynomial(x);

    std::vector<F> l = domain->EvaluateAllLagrangeCoefficients(x);

    // |num_qap_variables| = m = (l + 1 - 1) + m - l
    size_t num_qap_variables =
        (num_instance_variables - 1) + num_witness_variables;

    std::vector<F> a(num_qap_variables + 1);
    std::vector<F> b(num_qap_variables + 1);
    std::vector<F> c(num_qap_variables + 1);

    // clang-format off
    // |a[i]| = lₙ₊ᵢ(x) +  Σⱼ₌₀..ₙ₋₁ (lⱼ(x) * Aⱼ,ᵢ) (if i < |num_instance_variables|)
    //        = Σⱼ₌₀..ₙ₋₁ (lⱼ(x) * Aⱼ,ᵢ)            (otherwise)
    // |b[i]| = Σⱼ₌₀..ₙ₋₁ (lⱼ(x) * Bⱼ,ᵢ)
    // |c[i]| = Σⱼ₌₀..ₙ₋₁ (lⱼ(x) * Cⱼ,ᵢ)
    // clang-format on
    for (size_t i = 0; i < num_instance_variables; ++i) {
      a[i] = l[num_constraints + i];
    }

    for (size_t i = 0; i < num_constraints; ++i) {
      for (const Cell<F>& cell : matrices->a[i]) {
        a[cell.index] += (l[i] * cell.coefficient);
      }
      for (const Cell<F>& cell : matrices->b[i]) {
        b[cell.index] += (l[i] * cell.coefficient);
      }
      for (const Cell<F>& cell : matrices->c[i]) {
        c[cell.index] += (l[i] * cell.coefficient);
      }
    }

    return {std::move(a),          std::move(b),     std::move(c),
            std::move(t_x),        num_constraints,  num_instance_variables,
            num_witness_variables, num_qap_variables};
  }

  template <typename Domain, typename DensePoly = typename Domain::DensePoly>
  static WitnessMapResult<DensePoly> WitnessMap(
      const Domain* domain, const ConstraintSystem<F>& constraint_system) {
    std::optional<ConstraintMatrices<F>> matrices =
        constraint_system.ToMatrices();
    std::vector<F> full_assignments;
    full_assignments.reserve(constraint_system.num_instance_variables() +
                             constraint_system.num_constraints());
    full_assignments.insert(full_assignments.end(),
                            constraint_system.instance_assignments().begin(),
                            constraint_system.instance_assignments().end());
    full_assignments.insert(full_assignments.end(),
                            constraint_system.witness_assignments().begin(),
                            constraint_system.witness_assignments().end());

    DensePoly h_poly = WitnessMapFromMatrices(
        domain, matrices.value(), constraint_system.num_instance_variables(),
        constraint_system.num_constraints(), full_assignments);
    return {std::move(h_poly), std::move(full_assignments)};
  }

  template <typename Domain, typename DensePoly = typename Domain::DensePoly>
  static DensePoly WitnessMapFromMatrices(
      const Domain* domain, const ConstraintMatrices<F>& matrices,
      size_t num_instance_variables, size_t num_constraints,
      absl::Span<const F> full_assignments) {
    using Evals = typename Domain::Evals;

    std::vector<F> a(domain->size());
    std::vector<F> b(domain->size());
    std::vector<F> c(domain->size());

    // clang-format off
    // |a[i]| = Σⱼ₌₀..ₘ (xⱼ * Aᵢ,ⱼ)    (if i < |num_constraints|)
    //        = x[i - num_constraints] (otherwise)
    // |b[i]| = Σⱼ₌₀..ₘ (xⱼ * Bᵢ,ⱼ)    (if i < |num_constraints|)
    //        = 0                      (otherwise)
    // |c[i]| = Σⱼ₌₀..ₘ (xⱼ * Cᵢ,ⱼ)    (if i < |num_constraints|)
    //        = 0                      (otherwise)
    // where x is |full_assignments|.
    // clang-format on
    OPENMP_PARALLEL_FOR(size_t i = 0; i < num_constraints; ++i) {
      a[i] = EvaluateConstraint(matrices.a[i], full_assignments);
      b[i] = EvaluateConstraint(matrices.b[i], full_assignments);
      c[i] = EvaluateConstraint(matrices.c[i], full_assignments);
    }

    for (size_t i = num_constraints;
         i < num_constraints + num_instance_variables; ++i) {
      a[i] = full_assignments[i - num_constraints];
    }

    Evals a_evals(std::move(a));
    DensePoly a_poly = domain->IFFT(std::move(a_evals));
    Evals b_evals(std::move(b));
    DensePoly b_poly = domain->IFFT(std::move(b_evals));
    Evals c_evals(std::move(c));
    DensePoly c_poly = domain->IFFT(std::move(c_evals));

    std::unique_ptr<Domain> coset_domain =
        domain->GetCoset(F::FromMontgomery(F::Config::kSubgroupGenerator));

    a_evals = coset_domain->FFT(std::move(a_poly));
    b_evals = coset_domain->FFT(std::move(b_poly));
    c_evals = coset_domain->FFT(std::move(c_poly));

    Evals& ab_evals = a_evals *= b_evals;

    F vanishing_polynomial_over_coset =
        domain
            ->EvaluateVanishingPolynomial(
                F::FromMontgomery(F::Config::kSubgroupGenerator))
            .Inverse();

    // |h_evals[i]| = (|a[i]| * |b[i]| - |c[i]|)) / (g * ωⁿ⁺ˡ⁺¹ - 1)
    Evals& h_evals = ab_evals;
    OPENMP_PARALLEL_FOR(size_t i = 0; i < domain->size(); ++i) {
      F& h_evals_i = ab_evals.at(i);
      h_evals_i -= c_evals[i];
      h_evals_i *= vanishing_polynomial_over_coset;
    }

    return coset_domain->IFFT(std::move(h_evals));
  }

  template <typename Domain>
  static std::vector<F> ComputeHQuery(const Domain* domain, const F& t_x,
                                      const F& x, const F& delta_inverse) {
    std::vector<F> h_query(domain->size() - 1);
    base::Parallelize(h_query, [&t_x, &x, &delta_inverse](absl::Span<F> chunk,
                                                          size_t chunk_index,
                                                          size_t chunk_size) {
      size_t i = chunk_index * chunk_size;
      F x_i = x.Pow(i);
      for (F& v : chunk) {
        // (xⁱ * t(x)) / δ
        v = x_i;
        v *= t_x;
        v *= delta_inverse;
        x_i *= x;
      }
    });
    return h_query;
  }

 private:
  static F EvaluateConstraint(const std::vector<Cell<F>>& cells,
                              absl::Span<const F> assignments) {
    std::vector<F> sums = base::ParallelizeMap(
        cells, [assignments](absl::Span<const Cell<F>> chunk) {
          F sum;
          for (const Cell<F>& cell : chunk) {
            if (cell.coefficient.IsOne()) {
              sum += assignments[cell.index];
            } else {
              sum += assignments[cell.index] * cell.coefficient;
            }
          }
          return sum;
        });
    return std::accumulate(sums.begin(), sums.end(), F::Zero(), std::plus<>());
  }
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_
