#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/circuit/floor_planner/single_chip_layouter.h"

namespace tachyon::zk {

class SimpleFloorPlanner {
 public:
  template <typename F, typename Circuit, typename Config>
  static void Synthesize(Assignment<F>* assignment, Circuit& circuit,
                         Config&& config,
                         const std::vector<FixedColumnKey>& constants) {
    SingleChipLayouter layouter(assignment, constants);
    circuit.Synthesize(std::move(config), &layouter);
  }
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_
