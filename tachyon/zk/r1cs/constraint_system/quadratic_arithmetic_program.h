// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_

#include <stddef.h>

#include <memory>
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
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QUADRATIC_ARITHMETIC_PROGRAM_H_
