// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXAMPLES_SIMPLE_LOOKUP_CIRCUIT_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXAMPLES_SIMPLE_LOOKUP_CIRCUIT_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>

#include "tachyon/zk/plonk/circuit/circuit.h"

namespace tachyon::zk {

// This is taken and modified from
// https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/benches/dev_lookup.rs#L21-L26.
template <typename F, size_t Bits>
class SimpleLookupConfig {
 public:
  using Field = F;

  SimpleLookupConfig(const Selector& selector, const LookupTableColumn& table,
                     const AdviceColumnKey& advice)
      : selector_(selector), table_(table), advice_(advice) {}

  SimpleLookupConfig Clone() const {
    return SimpleLookupConfig(selector_, table_, advice_);
  }

  const Selector& selector() const { return selector_; }
  const LookupTableColumn& table() const { return table_; }
  const AdviceColumnKey& advice() const { return advice_; }

  void Load(Layouter<F>* layouter) const {
    layouter->AssignLookupTable(
        absl::Substitute("$0-bit table", Bits), [this](LookupTable<F>& table) {
          for (RowIndex row = 0; row < RowIndex{1} << Bits; ++row) {
            if (!table.AssignCell(
                    absl::Substitute("row $0", row), table_, row,
                    [row]() { return Value<F>::Known(F(row + 1)); }))
              return false;
          }
          return true;
        });
  }

 private:
  Selector selector_;
  LookupTableColumn table_;
  AdviceColumnKey advice_;
};

// This is taken and modified from
// https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/benches/dev_lookup.rs#L28-L91.
template <typename F, size_t Bits, template <typename> class _FloorPlanner>
class SimpleLookupCircuit : public Circuit<SimpleLookupConfig<F, Bits>> {
 public:
  using FloorPlanner =
      _FloorPlanner<SimpleLookupCircuit<F, Bits, _FloorPlanner>>;

  SimpleLookupCircuit() = default;
  explicit SimpleLookupCircuit(uint32_t k) : k_(k) {}

  std::unique_ptr<Circuit<SimpleLookupConfig<F, Bits>>> WithoutWitness()
      const override {
    return std::make_unique<SimpleLookupCircuit>();
  }

  static SimpleLookupConfig<F, Bits> Configure(ConstraintSystem<F>& meta) {
    SimpleLookupConfig<F, Bits> config(meta.CreateComplexSelector(),
                                       meta.CreateLookupTableColumn(),
                                       meta.CreateAdviceColumn());

    meta.Lookup("lookup", [&config](VirtualCells<F>& meta) {
      std::unique_ptr<Expression<F>> selector =
          meta.QuerySelector(config.selector());
      std::unique_ptr<Expression<F>> not_selector =
          ExpressionFactory<F>::Constant(F::One()) - selector->Clone();
      std::unique_ptr<Expression<F>> advice =
          meta.QueryAdvice(config.advice(), Rotation::Cur());

      LookupPairs<std::unique_ptr<Expression<F>>, LookupTableColumn>
          lookup_pairs;
      lookup_pairs.emplace_back(
          std::move(selector) * std::move(advice) + std::move(not_selector),
          config.table());
      return lookup_pairs;
    });

    return config;
  }

  void Synthesize(SimpleLookupConfig<F, Bits>&& config,
                  Layouter<F>* layouter) const override {
    config.Load(layouter);

    constexpr static size_t kModulus = size_t{1} << Bits;

    layouter->AssignRegion("assign values", [this, &config](Region<F>& region) {
      for (RowIndex offset = 0; offset < (RowIndex{1} << k_); ++offset) {
        config.selector().Enable(region, offset);
        region.AssignAdvice(
            absl::Substitute("offset $0", offset), config.advice(), offset,
            [offset]() { return Value<F>::Known(F(offset % kModulus + 1)); });
      }
    });
  }

 private:
  uint32_t k_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXAMPLES_SIMPLE_LOOKUP_CIRCUIT_H_
