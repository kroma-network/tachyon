// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_SELECTOR_DESCRIPTION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_SELECTOR_DESCRIPTION_H_

#include <utility>
#include <vector>

#include "tachyon/export.h"

namespace tachyon::zk {

/// This describes a selector and where it is activated.
class TACHYON_EXPORT SelectorDescription {
 public:
  SelectorDescription() = default;
  SelectorDescription(size_t selector_index, std::vector<bool>&& activations,
                      size_t max_degree)
      : selector_index_(selector_index),
        activations_(std::move(activations)),
        max_degree_(max_degree) {}

  size_t selector_index() const { return selector_index_; }
  const std::vector<bool>& activations() const { return activations_; }
  size_t max_degree() const { return max_degree_; }

 private:
  // The selector that this description references, by index.
  size_t selector_index_ = 0;
  // The vector of booleans defining which rows are active for this selector.
  std::vector<bool> activations_;
  // The maximum degree of a gate involving this selector, including the
  // virtual selector itself. This means this will be at least 1 for any
  // expression containing a simple selector, even if that selector is not
  // multiplied by anything.
  size_t max_degree_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_SELECTOR_DESCRIPTION_H_
