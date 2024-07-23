// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_

#include <stddef.h>

#include <functional>
#include <memory>
#include <memory_resource>
#include <optional>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/base/optional.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/zk/r1cs/constraint_system/constraint_system.h"
#include "tachyon/zk/r1cs/constraint_system/qap_instance_map_result.h"
#include "tachyon/zk/r1cs/constraint_system/qap_witness_map_result.h"

namespace tachyon::zk::r1cs {

template <typename F>
F EvaluateConstraint(const std::vector<Cell<F>>& cells,
                     absl::Span<const F> assignments) {
  F sum;
  for (const Cell<F>& cell : cells) {
    if (cell.coefficient.IsOne()) {
      sum += assignments[cell.index];
    } else {
      sum += assignments[cell.index] * cell.coefficient;
    }
  }
  return sum;
}

template <typename F>
class QuadraticArithmeticProgram {
 public:
  QuadraticArithmeticProgram() = delete;

  // Computes a QAP instance corresponding to the R1CS instance defined by |cs|.
  template <typename Domain>
  static QAPInstanceMapResult<F> InstanceMap(const Domain* domain,
                                             const ConstraintSystem<F>& cs,
                                             const F& x) {
    CHECK_GE(domain->size(), cs.num_constraints());
    std::optional<ConstraintMatrices<F>> matrices = cs.ToMatrices();
    // |num_constraint| = n
    size_t num_constraints = cs.num_constraints();
    // |num_instance_variables| = l + 1
    size_t num_instance_variables = cs.num_instance_variables();
    // |num_witness_variables| = m - l
    size_t num_witness_variables = cs.num_witness_variables();

    // t(x) = xⁿ⁺ˡ⁺¹ - 1
    F t_x = domain->EvaluateVanishingPolynomial(x);

    std::pmr::vector<F> l = domain->EvaluateAllLagrangeCoefficients(x);

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

  template <typename Domain>
  static QAPWitnessMapResult<F> WitnessMap(
      const Domain* domain, const ConstraintSystem<F>& constraint_system) {
    std::optional<ConstraintMatrices<F>> matrices =
        constraint_system.ToMatrices();
    std::vector<F> full_assignments;
    full_assignments.reserve(constraint_system.num_instance_variables() +
                             constraint_system.num_witness_variables());
    full_assignments.insert(full_assignments.end(),
                            constraint_system.instance_assignments().begin(),
                            constraint_system.instance_assignments().end());
    full_assignments.insert(full_assignments.end(),
                            constraint_system.witness_assignments().begin(),
                            constraint_system.witness_assignments().end());

    std::pmr::vector<F> h_poly =
        WitnessMapFromMatrices(domain, matrices.value(), full_assignments);
    return {std::move(h_poly), std::move(full_assignments)};
  }

  template <typename Domain>
  static std::pmr::vector<F> WitnessMapFromMatrices(
      const Domain* domain, const ConstraintMatrices<F>& matrices,
      absl::Span<const F> full_assignments) {
    using Evals = typename Domain::Evals;
    using DensePoly = typename Domain::DensePoly;

    CHECK_GE(domain->size(), matrices.num_constraints);

    std::vector<Evals> abc;
    abc.reserve(3);
    abc.emplace_back(std::pmr::vector<F>(domain->size()));
    abc.emplace_back(std::pmr::vector<F>(domain->size()));
    abc.emplace_back(std::pmr::vector<F>(domain->size()));

    // clang-format off
    // |a[i]| = Σⱼ₌₀..ₘ (xⱼ * Aᵢ,ⱼ)    (if i < |num_constraints|)
    //        = x[i - num_constraints] (otherwise)
    // |b[i]| = Σⱼ₌₀..ₘ (xⱼ * Bᵢ,ⱼ)    (if i < |num_constraints|)
    //        = 0                      (otherwise)
    // |c[i]| = Σⱼ₌₀..ₘ (xⱼ * Cᵢ,ⱼ)    (if i < |num_constraints|)
    //        = 0                      (otherwise)
    // where x is |full_assignments|.
    // clang-format on
    OMP_PARALLEL {
      OMP_FOR_NOWAIT
      for (size_t i = 0; i < matrices.num_constraints; ++i) {
        abc[0].evaluations()[i] =
            EvaluateConstraint(matrices.a[i], full_assignments);
      }

      OMP_FOR_NOWAIT
      for (size_t i = 0; i < matrices.num_constraints; ++i) {
        abc[1].evaluations()[i] =
            EvaluateConstraint(matrices.b[i], full_assignments);
      }

      OMP_FOR
      for (size_t i = 0; i < matrices.num_constraints; ++i) {
        abc[2].evaluations()[i] =
            EvaluateConstraint(matrices.c[i], full_assignments);
      }
    }

    for (size_t i = matrices.num_constraints;
         i < matrices.num_constraints + matrices.num_instance_variables; ++i) {
      abc[0].evaluations()[i] = full_assignments[i - matrices.num_constraints];
    }

    std::vector<DensePoly> abc_polys = domain->IFFT(std::move(abc));

    std::unique_ptr<Domain> coset_domain =
        domain->GetCoset(F::FromMontgomery(F::Config::kSubgroupGenerator));

    std::vector<Evals> abc_evals = coset_domain->FFT(std::move(abc_polys));

    F vanishing_polynomial_over_coset =
        unwrap(domain
                   ->EvaluateVanishingPolynomial(
                       F::FromMontgomery(F::Config::kSubgroupGenerator))
                   .Inverse());

    // |h_evals[i]| = (|a[i]| * |b[i]| - |c[i]|)) / (g * ωⁿ⁺ˡ⁺¹ - 1)
    OPENMP_PARALLEL_FOR(size_t i = 0; i < domain->size(); ++i) {
      F& h_evals_i = abc_evals[0].at(i);
      h_evals_i *= abc_evals[1][i];
      h_evals_i -= abc_evals[2][i];
      h_evals_i *= vanishing_polynomial_over_coset;
    }

    return coset_domain->IFFT(std::move(abc_evals[0]))
        .TakeCoefficients()
        .TakeCoefficients();
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

      // NOTE: It is not possible to have empty chunk so this is safe
      for (size_t i = 0; i < chunk.size() - 1; ++i) {
        // (xⁱ * t(x)) / δ
        chunk[i] = x_i * t_x;
        chunk[i] *= delta_inverse;
        x_i *= x;
      }
      chunk.back() = std::move(x_i);
      chunk.back() *= t_x;
      chunk.back() *= delta_inverse;
    });
    return h_query;
  }
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_
