#ifndef TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI2_CIRCUIT_H_
#define TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI2_CIRCUIT_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/constraint_system/circuit.h"

namespace tachyon::zk::plonk {

template <typename F>
struct Fibonacci2Config {
  using Field = F;

  Fibonacci2Config(const AdviceColumnKey& advice, Selector selector,
                   const InstanceColumnKey& instance)
      : advice(advice), selector(selector), instance(instance) {}

  Fibonacci2Config Clone() const { return {advice, selector, instance}; }

  AdviceColumnKey advice;
  Selector selector;
  InstanceColumnKey instance;
};

template <typename F>
class Fibonacci2Chip {
 public:
  explicit Fibonacci2Chip(const Fibonacci2Config<F>& config)
      : config_(config) {}

  static Fibonacci2Config<F> Configure(ConstraintSystem<F>& meta,
                                       const AdviceColumnKey& advice,
                                       const InstanceColumnKey& instance) {
    Selector selector = meta.CreateSimpleSelector();

    meta.EnableEquality(advice);
    meta.EnableEquality(instance);

    meta.CreateGate("add", [selector, &advice](VirtualCells<F>& meta) {
      //
      // advice | selector
      //   a    |   s
      //   b    |
      //   c    |
      //
      std::unique_ptr<Expression<F>> s = meta.QuerySelector(selector);
      std::unique_ptr<Expression<F>> a =
          meta.QueryAdvice(advice, Rotation::Cur());
      std::unique_ptr<Expression<F>> b =
          meta.QueryAdvice(advice, Rotation::Next());
      std::unique_ptr<Expression<F>> c = meta.QueryAdvice(advice, Rotation(2));
      std::vector<Constraint<F>> constraints;
      constraints.emplace_back(std::move(s) *
                               (std::move(a) + std::move(b) - std::move(c)));
      return constraints;
    });

    return Fibonacci2Config<F>(advice, selector, instance);
  }

  AssignedCell<F> Assign(Layouter<F>* layouter, RowIndex n_rows) const {
    AssignedCell<F> ret;
    layouter->AssignRegion(
        "entire fibonacci table", [this, &ret, n_rows](Region<F>& region) {
          config_.selector.Enable(region, 0);
          config_.selector.Enable(region, 1);

          AssignedCell<F> a_cell = region.AssignAdviceFromInstance(
              "1", config_.instance, 0, config_.advice, 0);
          AssignedCell<F> b_cell = region.AssignAdviceFromInstance(
              "1", config_.instance, 1, config_.advice, 1);

          for (RowIndex row = 2; row < n_rows; ++row) {
            if (row < n_rows - 2) {
              config_.selector.Enable(region, row);
            }

            const AssignedCell<F> c_cell = region.AssignAdvice(
                "advice", config_.advice, row, [&a_cell, &b_cell]() {
                  return a_cell.value() + b_cell.value();
                });

            a_cell = b_cell;
            b_cell = c_cell;
          }

          ret = b_cell;
        });
    return ret;
  }

  void ExposePublic(Layouter<F>* layouter, const AssignedCell<F>& cell,
                    RowIndex row) const {
    layouter->ConstrainInstance(cell.cell(), config_.instance, row);
  }

 private:
  Fibonacci2Config<F> config_;
};

template <typename F, template <typename> class _FloorPlanner>
class Fibonacci2Circuit : public Circuit<Fibonacci2Config<F>> {
 public:
  using FloorPlanner = _FloorPlanner<Fibonacci2Circuit<F, _FloorPlanner>>;

  std::unique_ptr<Circuit<Fibonacci2Config<F>>> WithoutWitness()
      const override {
    return std::make_unique<Fibonacci2Circuit>();
  }

  static Fibonacci2Config<F> Configure(ConstraintSystem<F>& meta) {
    AdviceColumnKey advice = meta.CreateAdviceColumn();
    InstanceColumnKey instance = meta.CreateInstanceColumn();
    return Fibonacci2Chip<F>::Configure(meta, advice, instance);
  }

  void Synthesize(Fibonacci2Config<F>&& config,
                  Layouter<F>* layouter) const override {
    Fibonacci2Chip<F> fibonacci2_chip(std::move(config));

    AssignedCell<F> out_cell =
        fibonacci2_chip.Assign(layouter->Namespace("entire table").get(), 10);

    fibonacci2_chip.ExposePublic(layouter->Namespace("out").get(), out_cell, 2);
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI2_CIRCUIT_H_
