#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/circuit/floor_planner/floor_planner.h"
#include "tachyon/zk/plonk/circuit/floor_planner/single_chip_layouter.h"

namespace tachyon::zk {

template <typename CircuitTy>
class SimpleFloorPlanner : public FloorPlanner<CircuitTy> {
 public:
  using F = typename CircuitTy::Field;
  using Config = typename CircuitTy::Config;

  void Synthesize(Assignment<F>* assignment, CircuitTy& circuit,
                  Config&& config,
                  const std::vector<FixedColumnKey>& constants) override {
    SingleChipLayouter layouter(assignment, constants);
    circuit.Synthesize(std::move(config), &layouter);
  }
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_
