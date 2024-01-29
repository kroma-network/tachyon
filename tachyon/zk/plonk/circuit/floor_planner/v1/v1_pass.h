// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_V1_PASS_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_V1_PASS_H_

#include <string>
#include <utility>
#include <variant>

#include "tachyon/zk/base/value.h"
#include "tachyon/zk/plonk/circuit/floor_planner/v1/assignment_pass.h"
#include "tachyon/zk/plonk/circuit/floor_planner/v1/measurement_pass.h"
#include "tachyon/zk/plonk/circuit/layouter.h"

namespace tachyon::zk {

// A single pass of the |V1FloorPlanner| layouter.
template <typename F>
class V1Pass : public Layouter<F> {
 public:
  using AssignRegionCallback = typename Layouter<F>::AssignRegionCallback;
  using AssignLookupTableCallback =
      typename Layouter<F>::AssignLookupTableCallback;

  explicit V1Pass(MeasurementPass<F>* measure) : pass_(measure) {}
  explicit V1Pass(AssignmentPass<F>* assign) : pass_(assign) {}

  // Layouter<F> methods
  void AssignRegion(std::string_view name,
                    AssignRegionCallback assign) override {
    std::visit([&](auto pass) { pass->AssignRegion(name, assign); }, pass_);
  }

  void AssignLookupTable(std::string_view name,
                         AssignLookupTableCallback assign) override {
    if (std::holds_alternative<AssignmentPass<F>*>(pass_)) {
      std::get<AssignmentPass<F>*>(pass_)->AssignLookupTable(name,
                                                             std::move(assign));
    }
  }

  void ConstrainInstance(const Cell& cell, const InstanceColumnKey& instance,
                         RowIndex row) override {
    if (std::holds_alternative<AssignmentPass<F>*>(pass_)) {
      std::get<AssignmentPass<F>*>(pass_)->ConstrainInstance(cell, instance,
                                                             row);
    }
  }

  Value<F> GetChallenge(const Challenge& challenge) const override {
    if (std::holds_alternative<MeasurementPass<F>*>(pass_)) {
      return Value<F>::Unknown();
    } else {
      return std::get<AssignmentPass<F>*>(pass_)
          ->plan()
          ->assignment()
          ->GetChallenge(challenge);
    }
  }

  Layouter<F>* GetRoot() override { return this; }

  void PushNamespace(std::string_view name) override {
    if (std::holds_alternative<AssignmentPass<F>*>(pass_)) {
      std::get<AssignmentPass<F>*>(pass_)->plan()->assignment()->PushNamespace(
          name);
    }
  }

  void PopNamespace(const std::optional<std::string>& gadget_name) override {
    if (std::holds_alternative<AssignmentPass<F>*>(pass_)) {
      std::get<AssignmentPass<F>*>(pass_)->plan()->assignment()->PopNamespace(
          gadget_name);
    }
  }

 private:
  std::variant<MeasurementPass<F>*, AssignmentPass<F>*> pass_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_V1_PASS_H_
