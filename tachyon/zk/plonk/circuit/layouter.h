// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_LAYOUTER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_LAYOUTER_H_

#include <memory>
#include <optional>
#include <string>

#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/plonk/circuit/challenge.h"
#include "tachyon/zk/plonk/circuit/region.h"
#include "tachyon/zk/plonk/circuit/table.h"
#include "tachyon/zk/plonk/constraint_system.h"
#include "tachyon/zk/plonk/error.h"

namespace tachyon::zk {

// A layout strategy within a circuit. The layouter is chip-agnostic and applies
// its strategy to the context and config it is given.
//
// This abstracts over the circuit assignments, handling row indices etc.
template <typename F>
class Layouter {
 public:
  using AnnotateCallback = base::OnceCallback<std::string()>;
  using AssignRegionCallback = base::RepeatingCallback<Error(Region<F>&)>;
  using AssignTableCallback = base::OnceCallback<Error(Table<F>&)>;
  using NameCallback = base::RepeatingCallback<std::string()>;

  virtual ~Layouter() = default;

  // Assign a region of gates to an absolute row number.
  //
  // Inside the closure, the chip may freely use relative offsets; the
  // |Layouter| will treat these assignments as a single "region" within the
  // circuit. Outside this closure, the |Layouter| is allowed to optimize as it
  // sees fit.
  virtual Error AssignRegion(NameCallback name, AssignRegionCallback assign) {
    return Error::kNone;
  }

  // Assign a table region to an absolute row number.
  virtual Error AssignTable(NameCallback name, AssignTableCallback assign) {
    return Error::kNone;
  }

  // Constrains a |cell| to equal an instance |column|'s row value at an
  // absolute position.
  virtual Error ConstrainInstance(const Cell& cell,
                                  const InstanceColumn& column, size_t row) {
    return Error::kNone;
  }

  // Queries the value of the given challenge.
  //
  // Returns |Value<F>::Unknown()| if the current synthesis phase is before the
  // challenge can be queried.
  virtual Value<F> GetChallenge(const Challenge& challenge) const {
    return Value<F>::Unknown();
  }

  // Gets the "root" of this assignment, bypassing the namespacing.
  //
  // TODO(chokobole): Update comment when NamespacedLayouter comes in.
  // Not intended for downstream consumption; use |Layouter::Namespace()|
  // instead.
  virtual Layouter<F>* GetRoot() { return nullptr; }

  // Creates a new (sub)namespace and enters into it.
  //
  // TODO(chokobole): Update comment when NamespacedLayouter comes in.
  // Not intended for downstream consumption; use |Layouter::Namespace()|
  // instead.
  virtual void PushNamespace(NameCallback name) {}

  // Exits out of the existing namespace.
  //
  // TODO(chokobole): Update comment when NamespacedLayouter comes in.
  // Not intended for downstream consumption; use |Layouter::Namespace()|
  // instead.
  virtual void PopNamespace(const std::optional<std::string>& gadget_name) {}
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_LAYOUTER_H_
