#ifndef TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI4_CIRCUIT_H_
#define TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI4_CIRCUIT_H_

#include <stddef.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/base/row_index.h"
#include "tachyon/zk/plonk/constraint_system/circuit.h"

namespace tachyon::zk::plonk {

template <typename F>
struct Fibonacci4Config {
  using Field = F;

  Fibonacci4Config(const std::array<AdviceColumnKey, 3>& advice,
                   const Selector& s_add, const Selector& s_xor,
                   const std::array<LookupTableColumn, 3>& xor_table,
                   const InstanceColumnKey& instance)
      : advice(advice),
        s_add(s_add),
        s_xor(s_xor),
        xor_table(xor_table),
        instance(instance) {}
  Fibonacci4Config(std::array<AdviceColumnKey, 3>&& advice,
                   const Selector& s_add, const Selector& s_xor,
                   std::array<LookupTableColumn, 3>&& xor_table,
                   const InstanceColumnKey& instance)
      : advice(std::move(advice)),
        s_add(s_add),
        s_xor(s_xor),
        xor_table(std::move(xor_table)),
        instance(instance) {}

  Fibonacci4Config Clone() const {
    return Fibonacci4Config(advice, s_add, s_xor, xor_table, instance);
  }

  std::array<AdviceColumnKey, 3> advice;
  Selector s_add;
  Selector s_xor;
  std::array<LookupTableColumn, 3> xor_table;
  InstanceColumnKey instance;
};

template <typename F>
class Fibonacci4Chip {
 public:
  explicit Fibonacci4Chip(Fibonacci4Config<F>&& config)
      : config_(std::move(config)) {}

  static Fibonacci4Config<F> Configure(ConstraintSystem<F>& meta) {
    std::array<AdviceColumnKey, 3> advice = {
        meta.CreateAdviceColumn(),
        meta.CreateAdviceColumn(),
        meta.CreateAdviceColumn(),
    };
    Selector s_add = meta.CreateSimpleSelector();
    Selector s_xor = meta.CreateComplexSelector();
    std::array<LookupTableColumn, 3> xor_table = {
        meta.CreateLookupTableColumn(),
        meta.CreateLookupTableColumn(),
        meta.CreateLookupTableColumn(),
    };
    InstanceColumnKey instance = meta.CreateInstanceColumn();
    Fibonacci4Config<F> config(std::move(advice), s_add, s_xor,
                               std::move(xor_table), instance);

    for (const AdviceColumnKey& advice : config.advice) {
      meta.EnableEquality(advice);
    }
    meta.EnableEquality(config.instance);

    meta.CreateGate("add", [&config](VirtualCells<F>& meta) {
      //
      // advice[0] | advice[1] | advice[2] | s_add
      //    a           b           c          s
      //
      std::unique_ptr<Expression<F>> s = meta.QuerySelector(config.s_add);
      std::unique_ptr<Expression<F>> a =
          meta.QueryAdvice(config.advice[0], Rotation::Cur());
      std::unique_ptr<Expression<F>> b =
          meta.QueryAdvice(config.advice[1], Rotation::Cur());
      std::unique_ptr<Expression<F>> c =
          meta.QueryAdvice(config.advice[2], Rotation::Cur());

      std::vector<Constraint<F>> constraints;
      constraints.emplace_back(std::move(s) *
                               (std::move(a) + std::move(b) - std::move(c)));
      return constraints;
    });

    meta.Lookup("lookup", [&config](VirtualCells<F>& meta) {
      std::unique_ptr<Expression<F>> s = meta.QuerySelector(config.s_xor);
      std::unique_ptr<Expression<F>> lhs =
          meta.QueryAdvice(config.advice[0], Rotation::Cur());
      std::unique_ptr<Expression<F>> rhs =
          meta.QueryAdvice(config.advice[1], Rotation::Cur());
      std::unique_ptr<Expression<F>> out =
          meta.QueryAdvice(config.advice[2], Rotation::Cur());

      LookupPairs<std::unique_ptr<Expression<F>>, LookupTableColumn> ret;
      ret.emplace_back(std::move(s->Clone()) * std::move(lhs),
                       config.xor_table[0]);
      ret.emplace_back(std::move(s->Clone()) * std::move(rhs),
                       config.xor_table[1]);
      ret.emplace_back(std::move(s) * std::move(out), config.xor_table[2]);
      return ret;
    });

    return config;
  }

  void LoadTable(Layouter<F>* layouter) const {
    layouter->AssignLookupTable("xor_table", [this](LookupTable<F>& table) {
      RowIndex idx = 0;
      for (size_t lhs = 0; lhs < 32; ++lhs) {
        for (size_t rhs = 0; rhs < 32; ++rhs) {
          if (!table.AssignCell("lhs", config_.xor_table[0], idx,
                                [lhs]() { return Value<F>::Known(F(lhs)); }) ||
              !table.AssignCell("rhs", config_.xor_table[1], idx,
                                [rhs]() { return Value<F>::Known(F(rhs)); }) ||
              !table.AssignCell(
                  "lhs ^ rhs", config_.xor_table[2], idx++,
                  [lhs, rhs]() { return Value<F>::Known(F(lhs ^ rhs)); }))
            return false;
        }
      }
      return true;
    });
  }

  AssignedCell<F> Assign(Layouter<F>* layouter, RowIndex n_rows) const {
    AssignedCell<F> ret;
    layouter->AssignRegion(
        "entire circuit", [this, &ret, n_rows](Region<F>& region) {
          config_.s_add.Enable(region, 0);

          // assign first row
          AssignedCell<F> a_cell = region.AssignAdviceFromInstance(
              "1", config_.instance, 0, config_.advice[0], 0);
          AssignedCell<F> b_cell = region.AssignAdviceFromInstance(
              "1", config_.instance, 1, config_.advice[1], 0);
          AssignedCell<F> c_cell = region.AssignAdvice(
              "add", config_.advice[2], 0,
              [&a_cell, &b_cell]() { return a_cell.value() + b_cell.value(); });

          // assign the rest of rows
          for (RowIndex row = 1; row < n_rows; ++row) {
            b_cell.CopyAdvice("a", region, config_.advice[0], row);
            c_cell.CopyAdvice("b", region, config_.advice[1], row);

            AssignedCell<F> new_c_cell;
            if (row % 2 == 0) {
              config_.s_add.Enable(region, row);
              new_c_cell = region.AssignAdvice(
                  "advice", config_.advice[2], row, [&b_cell, &c_cell]() {
                    return b_cell.value() + c_cell.value();
                  });
            } else {
              config_.s_xor.Enable(region, row);
              new_c_cell = region.AssignAdvice(
                  "advice", config_.advice[2], row, [&b_cell, &c_cell]() {
                    return b_cell.value().template AndThen<F>(
                        [&b = c_cell.value().value()](const F& a) {
                          // TODO(TomTaehoonKim): refac by making get_lower_32
                          // function
                          uint64_t a_val = a.ToBigInt()[0] & 0xffffffffU;
                          uint64_t b_val = b.ToBigInt()[0] & 0xffffffffU;
                          return Value<F>::Known(F(a_val ^ b_val));
                        });
                  });
            }

            b_cell = c_cell;
            c_cell = new_c_cell;
          }
          ret = std::move(c_cell);
        });
    return ret;
  }

  void ExposePublic(Layouter<F>* layouter, const AssignedCell<F>& cell,
                    RowIndex row) const {
    layouter->ConstrainInstance(cell.cell(), config_.instance, row);
  }

 private:
  Fibonacci4Config<F> config_;
};

template <typename F, template <typename> class _FloorPlanner>
class Fibonacci4Circuit : public Circuit<Fibonacci4Config<F>> {
 public:
  using FloorPlanner = _FloorPlanner<Fibonacci4Circuit<F, _FloorPlanner>>;

  std::unique_ptr<Circuit<Fibonacci4Config<F>>> WithoutWitness()
      const override {
    return std::make_unique<Fibonacci4Circuit>();
  }

  static Fibonacci4Config<F> Configure(ConstraintSystem<F>& meta) {
    return Fibonacci4Chip<F>::Configure(meta);
  }

  void Synthesize(Fibonacci4Config<F>&& config,
                  Layouter<F>* layouter) const override {
    Fibonacci4Chip<F> fibonacci4_chip(std::move(config));
    fibonacci4_chip.LoadTable(layouter->Namespace("lookup table").get());
    AssignedCell<F> out_cell =
        fibonacci4_chip.Assign(layouter->Namespace("entire table").get(), 8);
    fibonacci4_chip.ExposePublic(layouter->Namespace("out").get(), out_cell, 2);
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI4_CIRCUIT_H_
