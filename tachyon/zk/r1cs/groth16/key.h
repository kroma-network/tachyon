// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_GROTH16_KEY_H_
#define TACHYON_ZK_R1CS_GROTH16_KEY_H_

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/msm/fixed_base_msm.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/zk/r1cs/constraint_system/circuit.h"
#include "tachyon/zk/r1cs/constraint_system/qap_instance_map_result.h"
#include "tachyon/zk/r1cs/groth16/toxic_waste.h"

namespace tachyon::zk::r1cs::groth16 {

template <typename G1Point, size_t MaxDegree>
struct KeyPreLoadResult {
  using F = typename G1Point::ScalarField;

  std::unique_ptr<math::UnivariateEvaluationDomain<F, MaxDegree>> domain;
  QAPInstanceMapResult<F> qap_instance_map_result;
  math::FixedBaseMSM<G1Point> g1_msm;
  size_t non_zero_b;
};

class Key {
 protected:
  template <typename QAP, typename Curve, typename F, typename G1Point,
            size_t MaxDegree>
  void PreLoad(ToxicWaste<Curve>& toxic_waste, const Circuit<F>& circuit,
               KeyPreLoadResult<G1Point, MaxDegree>* result) {
    using Domain = math::UnivariateEvaluationDomain<F, MaxDegree>;

    ConstraintSystem<F> cs;
    cs.set_optimization_goal(OptimizationGoal::kConstraints);
    cs.set_mode(SynthesisMode::Setup());

    circuit.Synthesize(cs);

    cs.Finalize();

    std::unique_ptr<Domain> domain =
        Domain::Create(cs.num_constraints() + cs.num_instance_variables());

    toxic_waste.SampleX(domain.get());

    QAPInstanceMapResult<F> qap_instance_map_result =
        QAP::InstanceMap(domain.get(), cs, toxic_waste.x());

    size_t non_zero_a = CountNonZeros(qap_instance_map_result.num_qap_variables,
                                      qap_instance_map_result.a);
    size_t non_zero_b = CountNonZeros(qap_instance_map_result.num_qap_variables,
                                      qap_instance_map_result.b);

    result->g1_msm.Reset(non_zero_a + non_zero_b +
                             qap_instance_map_result.num_qap_variables +
                             domain->size() + 1,
                         toxic_waste.g1_generator());
    result->domain = std::move(domain);
    result->qap_instance_map_result = std::move(qap_instance_map_result);
    result->non_zero_b = non_zero_b;
  }

  template <typename F, typename Curve>
  F ComputeABC(const F& a, const F& b, const F& c,
               const ToxicWaste<Curve>& toxic_waste, const F& inverse) {
    F ret = toxic_waste.beta() * a;
    ret += toxic_waste.alpha() * b;
    ret += c;
    ret *= inverse;
    return ret;
  }

 private:
  template <typename F>
  static size_t CountNonZeros(size_t size, const std::vector<F>& vec) {
    absl::Span<const F> subspan = absl::MakeConstSpan(vec).subspan(0, size);
    std::vector<typename std::vector<F>::difference_type> results =
        base::ParallelizeMap(subspan, [](absl::Span<const F> chunk) {
          return std::count_if(chunk.begin(), chunk.end(),
                               [](const F& value) { return !value.IsZero(); });
        });
    return std::accumulate(results.begin(), results.end(), size_t{0},
                           std::plus<>());
  }
};

}  // namespace tachyon::zk::r1cs::groth16

#endif  // TACHYON_ZK_R1CS_GROTH16_KEY_H_
