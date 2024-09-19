// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_CUSTOM_GATE_EVALUATOR_H_
#define TACHYON_ZK_PLONK_VANISHING_CUSTOM_GATE_EVALUATOR_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/profiler.h"
#include "tachyon/zk/plonk/constraint_system/gate.h"
#include "tachyon/zk/plonk/vanishing/circuit_polynomial_builder_forward.h"
#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"

namespace tachyon::zk::plonk {

template <typename EvalsOrExtendedEvals>
class CustomGateEvaluator {
 public:
  using F = typename EvalsOrExtendedEvals::Field;

  void Construct(const std::vector<Gate<F>>& gates) {
    TRACE_EVENT("ProofGeneration",
                "Plonk::Vanishing::CustomGateEvaluator::Construct");
    std::vector<ValueSource> parts;
    for (const Gate<F>& gate : gates) {
      std::vector<ValueSource> tmp =
          base::Map(gate.polys(),
                    [this](const std::unique_ptr<Expression<F>>& expression) {
                      return evaluator_.AddExpression(expression.get());
                    });
      base::Extend(parts, std::move(tmp));
    }
    evaluator_.AddCalculation(Calculation::Horner(
        ValueSource::PreviousValue(), std::move(parts), ValueSource::Y()));
  }

  template <typename PS>
  void Evaluate(CircuitPolynomialBuilder<PS>& builder, absl::Span<F> chunk,
                size_t chunk_offset, size_t chunk_size) {
    TRACE_EVENT("ProofGeneration",
                "Plonk::Vanishing::CustomGateEvaluator::Evaluate");
    EvaluationInput<EvalsOrExtendedEvals> evaluation_input =
        builder.ExtractEvaluationInput(evaluator_.CreateInitialIntermediates(),
                                       evaluator_.CreateEmptyRotations());
    size_t start = chunk_offset * chunk_size;
    for (size_t i = 0; i < chunk.size(); ++i) {
      chunk[i] = evaluator_.Evaluate(evaluation_input, start + i,
                                     /*scale=*/1, chunk[i]);
    }
  }

  template <typename PS>
  void UpdateCosets(CircuitPolynomialBuilder<PS>& builder, size_t circuit_idx) {
    using PCS = typename PS::PCS;
    using PCS = typename PS::PCS;
    using Poly = typename PCS::Poly;

    TRACE_EVENT("ProofGeneration",
                "Plonk::Vanishing::CustomGateEvaluator::UpdateCosets");

    constexpr halo2::Vendor kVendor = PS::kVendor;

    if constexpr (kVendor == halo2::Vendor::kScroll) {
      TRACE_EVENT("Subtask", "Calculate fixed column cosets");
      absl::Span<const Poly> new_fixed_columns =
          builder.poly_tables_[circuit_idx].GetFixedColumns();
      fixed_column_cosets_.resize(new_fixed_columns.size());
      for (size_t i = 0; i < new_fixed_columns.size(); ++i) {
        fixed_column_cosets_[i] =
            builder.coset_domain_->FFT(new_fixed_columns[i]);
      }
    }

    {
      TRACE_EVENT("Subtask", "Calculate advice column cosets");
      absl::Span<const Poly> new_advice_columns =
          builder.poly_tables_[circuit_idx].GetAdviceColumns();
      advice_column_cosets_.resize(new_advice_columns.size());
      for (size_t i = 0; i < new_advice_columns.size(); ++i) {
        if constexpr (kVendor == halo2::Vendor::kPSE) {
          advice_column_cosets_[i] =
              CoeffToExtended(new_advice_columns[i], builder.extended_domain_);
        } else {
          advice_column_cosets_[i] =
              builder.coset_domain_->FFT(new_advice_columns[i]);
        }
      }
    }

    {
      TRACE_EVENT("Subtask", "Calculate instance column cosets");
      absl::Span<const Poly> new_instance_columns =
          builder.poly_tables_[circuit_idx].GetInstanceColumns();
      instance_column_cosets_.resize(new_instance_columns.size());
      for (size_t i = 0; i < new_instance_columns.size(); ++i) {
        if constexpr (kVendor == halo2::Vendor::kPSE) {
          instance_column_cosets_[i] = CoeffToExtended(
              new_instance_columns[i], builder.extended_domain_);
        } else {
          instance_column_cosets_[i] =
              builder.coset_domain_->FFT(new_instance_columns[i]);
        }
      }
    }

    if constexpr (kVendor == halo2::Vendor::kPSE) {
      TRACE_EVENT("Subtask", "Construct table PSE");
      builder.table_ = {
          builder.proving_key_.fixed_cosets(),
          advice_column_cosets_,
          instance_column_cosets_,
          builder.poly_tables_[circuit_idx].challenges(),
      };
    } else {
      TRACE_EVENT("Subtask", "Construct table Scroll");
      builder.table_ = {
          fixed_column_cosets_,
          advice_column_cosets_,
          instance_column_cosets_,
          builder.poly_tables_[circuit_idx].challenges(),
      };
    }
  }

 private:
  GraphEvaluator<F> evaluator_;
  std::vector<EvalsOrExtendedEvals> fixed_column_cosets_;
  std::vector<EvalsOrExtendedEvals> advice_column_cosets_;
  std::vector<EvalsOrExtendedEvals> instance_column_cosets_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_CUSTOM_GATE_EVALUATOR_H_
