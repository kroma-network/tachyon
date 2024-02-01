// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/cycle_store.h"

#include <utility>

namespace tachyon::zk::plonk {

bool CycleStore::MergeCycle(const Label& label, const Label& label2) {
  Label left_cycle_base = GetCycleBase(label);
  Label right_cycle_base = GetCycleBase(label2);
  if (left_cycle_base == right_cycle_base) return false;

  // Ensure that the cell with a larger cycle size becomes the left.
  if (sizes_[left_cycle_base] < sizes_[right_cycle_base]) {
    std::swap(left_cycle_base, right_cycle_base);
  }

  // Merge the right cycle into the left one.
  sizes_[left_cycle_base] += sizes_[right_cycle_base];
  Label l = right_cycle_base;
  while (true) {
    aux_[l] = left_cycle_base;
    l = mapping_[l];
    if (l == right_cycle_base) {
      break;
    }
  }

  std::swap(mapping_[label], mapping_[label2]);
  return true;
}

std::vector<Label> CycleStore::GetAllLabels(const Label& label) const {
  std::vector<Label> ret;
  Label base = GetCycleBase(label);
  Label l = base;
  ret.push_back(l);
  while (true) {
    l = mapping_[l];
    ret.push_back(l);
    if (l == base) {
      break;
    }
  }
  return ret;
}

}  // namespace tachyon::zk::plonk
