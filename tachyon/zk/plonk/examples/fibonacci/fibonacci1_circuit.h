#ifndef TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI1_CIRCUIT_H_
#define TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI1_CIRCUIT_H_

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/constraint_system/circuit.h"

namespace tachyon::zk::plonk {

template <typename F>
struct AssignedRow {
  AssignedCell<F> a;
  AssignedCell<F> b;
  AssignedCell<F> c;

  AssignedRow() = default;
  AssignedRow(AssignedCell<F>&& a, AssignedCell<F>&& b, AssignedCell<F>&& c)
      : a(std::move(a)), b(std::move(b)), c(std::move(c)) {}
};

template <typename F>
struct Fibonacci1Config {
  using Field = F;

  Fibonacci1Config(std::array<AdviceColumnKey, 3> advice, Selector selector,
                   const InstanceColumnKey& instance)
      : advice(advice), selector(selector), instance(instance) {}

  Fibonacci1Config Clone() const { return {advice, selector, instance}; }

  std::array<AdviceColumnKey, 3> advice;
  Selector selector;
  InstanceColumnKey instance;
};

template <typename F>
class Fibonacci1Chip {
 public:
  explicit Fibonacci1Chip(const Fibonacci1Config<F>& config)
      : config_(config) {}

  static Fibonacci1Config<F> Configure(ConstraintSystem<F>& meta) {
    std::array<AdviceColumnKey, 3> advice = {
        meta.CreateAdviceColumn(),
        meta.CreateAdviceColumn(),
        meta.CreateAdviceColumn(),
    };
    Selector selector = meta.CreateSimpleSelector();
    InstanceColumnKey instance = meta.CreateInstanceColumn();

    for (const AdviceColumnKey& column : advice) {
      meta.EnableEquality(column);
    }
    meta.EnableEquality(instance);

    meta.CreateGate("add", [selector, &advice](VirtualCells<F>& meta) {
      //
      // advice[0] | advice[1] | advice[2] | selector
      //    a           b           c           s
      //
      std::unique_ptr<Expression<F>> s = meta.QuerySelector(selector);
      std::unique_ptr<Expression<F>> a =
          meta.QueryAdvice(advice[0], Rotation::Cur());
      std::unique_ptr<Expression<F>> b =
          meta.QueryAdvice(advice[1], Rotation::Cur());
      std::unique_ptr<Expression<F>> c =
          meta.QueryAdvice(advice[2], Rotation::Cur());

      std::vector<Constraint<F>> constraints;
      constraints.emplace_back(std::move(s) *
                               (std::move(a) + std::move(b) - std::move(c)));
      return constraints;
    });

    return Fibonacci1Config<F>(std::move(advice), selector,
                               std::move(instance));
  }

  AssignedRow<F> AssignFirstRow(Layouter<F>* layouter) const {
    AssignedRow<F> ret;
    layouter->AssignRegion("first row", [this, &ret](Region<F>& region) {
      config_.selector.Enable(region, 0);

      AssignedCell<F> a_cell = region.AssignAdviceFromInstance(
          "f(0)", config_.instance, 0, config_.advice[0], 0);
      AssignedCell<F> b_cell = region.AssignAdviceFromInstance(
          "f(1)", config_.instance, 1, config_.advice[1], 0);
      AssignedCell<F> c_cell = region.AssignAdvice(
          "a + b", config_.advice[2], 0,
          [&a_cell, &b_cell]() { return a_cell.value() + b_cell.value(); });

      ret = AssignedRow<F>(std::move(a_cell), std::move(b_cell),
                           std::move(c_cell));
    });
    return ret;
  }

  AssignedCell<F> AssignRow(Layouter<F>* layouter,
                            const AssignedCell<F>& prev_b,
                            const AssignedCell<F>& prev_c) const {
    AssignedCell<F> ret;
    layouter->AssignRegion(
        "next row", [this, &ret, &prev_b, &prev_c](Region<F>& region) {
          config_.selector.Enable(region, 0);

          // Copy the value from b & c in previous row to a & b in current row
          const AssignedCell<F> a_cell =
              prev_b.CopyAdvice("a", region, config_.advice[0], 0);
          const AssignedCell<F> b_cell =
              prev_c.CopyAdvice("b", region, config_.advice[1], 0);

          const AssignedCell<F> c_cell = region.AssignAdvice(
              "a + b", config_.advice[2], 0,
              [&a_cell, &b_cell]() { return a_cell.value() + b_cell.value(); });

          ret = std::move(c_cell);
        });
    return ret;
  }

  void ExposePublic(Layouter<F>* layouter, const AssignedCell<F>& cell,
                    RowIndex row) const {
    layouter->ConstrainInstance(cell.cell(), config_.instance, row);
  }

 private:
  Fibonacci1Config<F> config_;
};

template <typename F, template <typename> class _FloorPlanner>
class Fibonacci1Circuit : public Circuit<Fibonacci1Config<F>> {
 public:
  using FloorPlanner = _FloorPlanner<Fibonacci1Circuit<F, _FloorPlanner>>;

  std::unique_ptr<Circuit<Fibonacci1Config<F>>> WithoutWitness()
      const override {
    return std::make_unique<Fibonacci1Circuit>();
  }

  static Fibonacci1Config<F> Configure(ConstraintSystem<F>& meta) {
    return Fibonacci1Chip<F>::Configure(meta);
  }

  void Synthesize(Fibonacci1Config<F>&& config,
                  Layouter<F>* layouter) const override {
    Fibonacci1Chip<F> fibonacci1_chip(std::move(config));

    AssignedRow<F> first_row =
        fibonacci1_chip.AssignFirstRow(layouter->Namespace("first row").get());

    AssignedCell<F> prev_b = first_row.b;
    AssignedCell<F> prev_c = first_row.c;

    for (RowIndex i = 3; i < 10; ++i) {
      AssignedCell<F> c_cell = fibonacci1_chip.AssignRow(
          layouter->Namespace("next row").get(), prev_b, prev_c);
      prev_b = prev_c;
      prev_c = c_cell;
    }

    fibonacci1_chip.ExposePublic(layouter->Namespace("out").get(), prev_c, 2);
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI1_CIRCUIT_H_
