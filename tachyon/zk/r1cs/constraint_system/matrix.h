// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_MATRIX_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_MATRIX_H_

#include <stddef.h>

#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"

namespace tachyon::zk::r1cs {

// FIXME(chokobole): I want to separate |coefficient| from |Cell|.
// See comments in tachyon/zk/r1cs/constraint_system/constraint_matrices.h.
template <typename F>
struct Cell {
  F coefficient;
  size_t index;

  bool operator==(const Cell& other) const {
    return coefficient == other.coefficient && index == other.index;
  }
  bool operator!=(const Cell& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("{coefficient: $0, index: $1}",
                            coefficient.ToString(), index);
  }
};

template <typename F>
class Matrix {
 public:
  Matrix() = default;
  explicit Matrix(const std::vector<std::vector<Cell<F>>>& cells)
      : cells_(cells) {}
  explicit Matrix(std::vector<std::vector<Cell<F>>>&& cells)
      : cells_(std::move(cells)) {}

  size_t CountNonZero() const {
    return std::accumulate(cells_.begin(), cells_.end(), 0,
                           [](size_t acc, const std::vector<Cell<F>>& row) {
                             return acc + row.size();
                           });
  }

  bool operator==(const Matrix& other) const { return cells_ == other.cells_; }
  bool operator!=(const Matrix& other) const { return cells_ != other.cells_; }

  std::string ToString() const {
    std::vector<std::string> rows =
        base::Map(cells_, [](const std::vector<Cell<F>>& row) {
          return absl::Substitute(
              "[$0]", absl::StrJoin(base::Map(row,
                                              [](const Cell<F>& cell) {
                                                return cell.ToString();
                                              }),
                                    ","));
        });
    return absl::Substitute("[$0]", absl::StrJoin(rows, ",\n"));
  }

 private:
  std::vector<std::vector<Cell<F>>> cells_;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_MATRIX_H_
