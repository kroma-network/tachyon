#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/layout/floor_planner/floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/single_chip_layouter.h"

namespace tachyon::zk {

template <typename Circuit>
class SimpleFloorPlanner : public FloorPlanner<Circuit> {
 public:
  using F = typename Circuit::Field;
  using Config = typename Circuit::Config;

  void Synthesize(Assignment<F>* assignment, const Circuit& circuit,
                  Config&& config,
                  const std::vector<FixedColumnKey>& constants) override {
    SingleChipLayouter layouter(assignment, constants);
    circuit.Synthesize(std::move(config), &layouter);
  }
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SIMPLE_FLOOR_PLANNER_H_
