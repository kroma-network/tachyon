// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_CYCLE_STORE_H_
#define TACHYON_ZK_PLONK_PERMUTATION_CYCLE_STORE_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/export.h"
#include "tachyon/zk/plonk/permutation/label.h"

namespace tachyon::zk::plonk {

// CycleStore stores cycle that is used to constrain equality between values.
//
//   For example, given two disjoint cycles (A B C D) and (E F G H):
//
//   A +---> B
//   ^       +
//   |       |
//   +       v
//   D <---+ C       E +---> F
//                   ^       +
//                   |       |
//                   +       v
//                   H <---+ G
//
//   CycleStore store;
//   Label a; // A
//   Label b; // B
//   Label e; // E
//
//   CHECK_EQ(store.GetCycleBase(a), store.GetCycleBase(b));
//   CHECK(store.CheckSameCycle(a, b));
//   CHECK_NE(store.GetCycleBase(a), store.GetCycleBase(e));
//   CHECK(!store.CheckSameCycle(a, e));
//
//   CHECK_EQ(store.GetCycleSize(a), size_t{4});
//
//   After adding constraint B ≡ E the above algorithm produces the cycle:
//
//   A +---> B +-------------+
//   ^                       |
//   |                       |
//   +                       v
//   D <---+ C <---+ E       F
//                   ^       +
//                   |       |
//                   +       v
//                   H <---+ G
//
//   CHECK_EQ(store.GetCycleBase(a), store.GetCycleBase(b));
//   CHECK(store.CheckSameCycle(a, b));
//   CHECK_EQ(store.GetCycleBase(a), store.GetCycleBase(e));
//   CHECK(store.CheckSameCycle(a, e));
//
//   CHECK_EQ(store.GetCycleSize(a), size_t{8});
//
// See
// https://zcash.github.io/halo2/design/proving-system/permutation.html#algorithm.
class TACHYON_EXPORT CycleStore {
 public:
  template <typename T>
  class Table {
   public:
    Table() = default;
    explicit Table(std::vector<std::vector<T>>&& values)
        : values_(std::move(values)) {}

    T& operator[](const Label& l) { return values_[l.col][l.row]; }
    const T& operator[](const Label& l) const { return values_[l.col][l.row]; }

    bool operator==(const Table<T>& other) const {
      return values_ == other.values_;
    }
    bool operator!=(const Table<T>& other) const {
      return values_ != other.values_;
    }

    bool IsEmpty() const { return values_.empty(); }

   private:
    std::vector<std::vector<T>> values_;
  };

  CycleStore() = default;
  CycleStore(size_t cols, RowIndex rows) {
    mapping_ = Table(base::CreateVector(cols, [rows](size_t col) {
      return base::CreateVector(
          rows, [col](RowIndex row) { return Label(col, row); });
    }));
    aux_ = mapping_;
    sizes_ = Table(std::vector<std::vector<size_t>>(
        cols, std::vector<size_t>(rows, size_t{1})));
  }

  const Table<Label>& mapping() const { return mapping_; }
  const Table<Label>& aux() const { return aux_; }
  const Table<size_t>& sizes() const { return sizes_; }

  // Return the next label of given |label| within a cycle.
  const Label& GetNextLabel(const Label& label) const {
    return mapping_[label];
  }

  // Return the representative of cycle of given |label|.
  const Label& GetCycleBase(const Label& label) const { return aux_[label]; }

  // Return the size of the representative of cycle of given |label|.
  size_t GetCycleSize(const Label& label) const {
    return sizes_[GetCycleBase(label)];
  }

  // Return whether the representative of cycles of given |label| and |label2|
  // belong to the same cycle.
  bool CheckSameCycle(const Label& label, const Label& label2) const {
    return GetCycleBase(label) == GetCycleBase(label2);
  }

  // Return false if the cycles of |label| and |label2| are same.
  // Return true if two cycles are merged.
  bool MergeCycle(const Label& label, const Label& label2);

  // Return all the labels that are belong to the cycle of given |label|.
  std::vector<Label> GetAllLabels(const Label& label) const;

 private:
  // |mapping_| keeps track of the next label for each cycle.
  Table<Label> mapping_;
  // |aux_| keeps track of the cycle for each label.
  Table<Label> aux_;
  // |sizes_| keeps track of the size of each cycle.
  Table<size_t> sizes_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_CYCLE_STORE_H_
