// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_NAMESPACED_LAYOUTER_H_
#define TACHYON_ZK_PLONK_LAYOUT_NAMESPACED_LAYOUTER_H_

#include <memory>
#include <string>
#include <utility>

#include "tachyon/zk/plonk/layout/layouter.h"

// This is a "namespaced" layouter which borrows a |Layouter| (pushing a
// namespace context) and, when dropped, pops out of the namespace context.
namespace tachyon::zk::plonk {

template <typename F>
class NamespacedLayouter : public Layouter<F> {
 public:
  using AssignRegionCallback = typename Layouter<F>::AssignRegionCallback;
  using AssignLookupTableCallback =
      typename Layouter<F>::AssignLookupTableCallback;

  explicit NamespacedLayouter(Layouter<F>* layouter) : layouter_(layouter) {}

  ~NamespacedLayouter() override {
    // TODO(chokobole): Understand why halo2 needs this
    // See
    // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/circuit.rs#L554-L579.
    layouter_->PopNamespace(std::nullopt);
  }

  // Layouter<F> methods
  void AssignRegion(std::string_view name,
                    AssignRegionCallback assign) override {
    layouter_->AssignRegion(name, std::move(assign));
  }

  void AssignLookupTable(std::string_view name,
                         AssignLookupTableCallback assign) override {
    layouter_->AssignLookupTable(name, std::move(assign));
  }

  void ConstrainInstance(const Cell& cell, const InstanceColumnKey& column,
                         RowIndex row) override {
    layouter_->ConstrainInstance(cell, column, row);
  }

  Value<F> GetChallenge(Challenge challenge) const override {
    return layouter_->GetChallenge(challenge);
  }

  Layouter<F>* GetRoot() override { return layouter_->GetRoot(); }

  void PushNamespace(std::string_view) override {
    NOTREACHED() << "Only the root layouter is allowed to call PushNamespace()";
  }

  void PopNamespace(const std::optional<std::string>&) override {
    NOTREACHED() << "Only the root layouter is allowed to call PopNamespace()";
  }

 private:
  // not owned
  Layouter<F>* const layouter_;
};

template <typename F>
std::unique_ptr<NamespacedLayouter<F>> Layouter<F>::Namespace(
    std::string_view name) {
  GetRoot()->PushNamespace(name);
  return std::make_unique<NamespacedLayouter<F>>(GetRoot());
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_NAMESPACED_LAYOUTER_H_
