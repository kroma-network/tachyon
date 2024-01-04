// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_POINT_SET_H_
#define TACHYON_ZK_BASE_POINT_SET_H_

#include <memory>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/ref.h"

namespace tachyon::zk {

template <typename Point>
class PointSet {
 public:
  PointSet() = default;
  explicit PointSet(size_t expected_size) { points_.reserve(expected_size); }

  const std::vector<std::unique_ptr<Point>>& points() const { return points_; }

  base::DeepRef<const Point> Insert(const Point& item) {
    std::optional<size_t> existing_idx =
        base::FindIndexIf(points_.begin(), points_.end(),
                          [&item](const std::unique_ptr<Point>& point) {
                            return *point == item;
                          });
    if (!existing_idx.has_value()) {
      existing_idx = points_.size();
      points_.push_back(std::make_unique<Point>(item));
    }
    return base::DeepRef<const Point>(points_[existing_idx.value()].get());
  }

 private:
  std::vector<std::unique_ptr<Point>> points_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_POINT_SET_H_
