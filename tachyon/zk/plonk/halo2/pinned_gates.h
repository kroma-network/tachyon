// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_PINNED_GATES_H_
#define TACHYON_ZK_PLONK_HALO2_PINNED_GATES_H_

#include <memory>
#include <string>
#include <vector>

#include "tachyon/zk/plonk/constraint_system/gate.h"
#include "tachyon/zk/plonk/halo2/stringifiers/expression_stringifier.h"

namespace tachyon {
namespace zk::halo2 {

template <typename F>
class PinnedGates {
 public:
  explicit PinnedGates(const std::vector<Gate<F>>& gates) : gates_(gates) {}

  const std::vector<Gate<F>>& gates() const { return gates_; }

 private:
  const std::vector<Gate<F>>& gates_;
};

}  // namespace zk::halo2

namespace base::internal {

template <typename F>
class RustDebugStringifier<zk::halo2::PinnedGates<F>> {
 public:
  static std::ostream& AppendToStream(
      std::ostream& os, RustFormatter& fmt,
      const zk::halo2::PinnedGates<F>& pinned_gates) {
    DebugList list = fmt.DebugList();
    for (const zk::Gate<F>& gate : pinned_gates.gates()) {
      const std::vector<std::unique_ptr<zk::Expression<F>>>& polys =
          gate.polys();
      for (const std::unique_ptr<zk::Expression<F>>& poly : polys) {
        list.Entry(*poly);
      }
    }
    return os << list.Finish();
  }
};

}  // namespace base::internal
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_PINNED_GATES_H_
