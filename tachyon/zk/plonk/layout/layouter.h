// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_LAYOUTER_H_
#define TACHYON_ZK_PLONK_LAYOUT_LAYOUTER_H_

#include <memory>
#include <optional>
#include <string>

#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/base/row_index.h"
#include "tachyon/zk/plonk/constraint_system/challenge.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/layout/lookup_table.h"
#include "tachyon/zk/plonk/layout/region.h"

namespace tachyon::zk::plonk {

template <typename F>
class NamespacedLayouter;

// A layout strategy within a circuit. The layouter is chip-agnostic and applies
// its strategy to the context and config it is given.
//
// This abstracts over the circuit assignments, handling row indices etc.
template <typename F>
class Layouter {
 public:
  using AssignRegionCallback = base::RepeatingCallback<void(Region<F>&)>;
  using AssignLookupTableCallback = base::OnceCallback<void(LookupTable<F>&)>;

  virtual ~Layouter() = default;

  // Assign a region of gates to an absolute row number.
  //
  // Inside the closure, the chip may freely use relative offsets; the
  // |Layouter| will treat these assignments as a single "region" within the
  // circuit. Outside this closure, the |Layouter| is allowed to optimize as it
  // sees fit.
  virtual void AssignRegion(std::string_view name,
                            AssignRegionCallback assign) = 0;

  // Assign a table region to an absolute row number.
  virtual void AssignLookupTable(std::string_view name,
                                 AssignLookupTableCallback assign) = 0;

  // Constrains a |cell| to equal an instance |column|'s row value at an
  // absolute position.
  virtual void ConstrainInstance(const Cell& cell,
                                 const InstanceColumnKey& column,
                                 RowIndex row) = 0;

  // Queries the value of the given challenge.
  //
  // Returns |Value<F>::Unknown()| if the current synthesis phase is before the
  // challenge can be queried.
  virtual Value<F> GetChallenge(const Challenge& challenge) const = 0;

  // Gets the "root" of this assignment, bypassing the namespacing.
  //
  // Not intended for downstream consumption; use |Layouter::Namespace()|
  // instead.
  virtual Layouter<F>* GetRoot() = 0;

  // Creates a new (sub)namespace and enters into it.
  //
  // Not intended for downstream consumption; use |Layouter::Namespace()|
  // instead.
  virtual void PushNamespace(std::string_view name) = 0;

  // Exits out of the existing namespace.
  //
  // Not intended for downstream consumption; use |Layouter::Namespace()|
  // instead.
  virtual void PopNamespace(const std::optional<std::string>& gadget_name) = 0;

  // Enters into a namespace
  std::unique_ptr<NamespacedLayouter<F>> Namespace(std::string_view name);
};

}  // namespace tachyon::zk::plonk

#include "tachyon/zk/plonk/layout/namespaced_layouter.h"

#endif  // TACHYON_ZK_PLONK_LAYOUT_LAYOUTER_H_
