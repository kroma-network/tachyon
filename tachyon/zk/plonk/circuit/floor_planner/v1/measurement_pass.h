// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_MEASUREMENT_PASS_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_MEASUREMENT_PASS_H_

#include <string>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/circuit/layouter.h"
#include "tachyon/zk/plonk/circuit/region_shape.h"

namespace tachyon::zk {

// Measures the circuit.
template <typename F>
class MeasurementPass {
 public:
  using AssignRegionCallback = typename Layouter<F>::AssignRegionCallback;

  struct Region {
    Region(std::string_view name, RegionShape<F>&& shape)
        : name(std::string(name)), shape(std::move(shape)) {}

    std::string name;
    RegionShape<F> shape;
  };

  MeasurementPass() = default;

  const std::vector<Region>& regions() const { return regions_; }

  void AssignRegion(std::string_view name, AssignRegionCallback assign) {
    size_t region_index = regions_.size();

    // Get shape of the region.
    RegionShape<F> shape(region_index);
    {
      zk::Region<F> region(&shape);
      assign.Run(region);
    }
    regions_.emplace_back(name, std::move(shape));
  }

 private:
  std::vector<Region> regions_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_MEASUREMENT_PASS_H_
