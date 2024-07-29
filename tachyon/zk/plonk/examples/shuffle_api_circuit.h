// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

// This is taken and modified from
// https://github.com/privacy-scaling-explorations/halo2/blob/bc857a7/halo2_proofs/tests/shuffle_api.rs.

#ifndef TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_API_CIRCUIT_H_
#define TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_API_CIRCUIT_H_

#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/constraint_system/circuit.h"

namespace tachyon::zk::plonk {

template <typename F>
class ShuffleAPIConfig {
 public:
  using Field = F;

  ShuffleAPIConfig() = default;
  ShuffleAPIConfig(const AdviceColumnKey& input_0,
                   const FixedColumnKey& input_1,
                   const AdviceColumnKey& shuffle_0,
                   const AdviceColumnKey& shuffle_1, Selector s_input,
                   Selector s_shuffle)
      : input_0_(input_0),
        input_1_(input_1),
        shuffle_0_(shuffle_0),
        shuffle_1_(shuffle_1),
        s_input_(s_input),
        s_shuffle_(s_shuffle) {}

  ShuffleAPIConfig Clone() const {
    return {input_0_, input_1_, shuffle_0_, shuffle_1_, s_input_, s_shuffle_};
  }

  const AdviceColumnKey& input_0() const { return input_0_; }
  const FixedColumnKey& input_1() const { return input_1_; }
  const AdviceColumnKey& shuffle_0() const { return shuffle_0_; }
  const AdviceColumnKey& shuffle_1() const { return shuffle_1_; }
  Selector s_input() const { return s_input_; }
  Selector s_shuffle() const { return s_shuffle_; }

  static ShuffleAPIConfig Configure(ConstraintSystem<F>& meta,
                                    const AdviceColumnKey& input_0,
                                    const FixedColumnKey& input_1,
                                    const AdviceColumnKey& shuffle_0,
                                    const AdviceColumnKey& shuffle_1) {
    Selector s_shuffle = meta.CreateComplexSelector();
    Selector s_input = meta.CreateComplexSelector();

    meta.Shuffle("shuffle", [s_shuffle, s_input, &input_0, &input_1, &shuffle_0,
                             &shuffle_1](VirtualCells<F>& meta) {
      std::unique_ptr<Expression<F>> s_input_expr = meta.QuerySelector(s_input);
      std::unique_ptr<Expression<F>> s_shuffle_expr =
          meta.QuerySelector(s_shuffle);
      std::unique_ptr<Expression<F>> input_0_expr =
          meta.QueryAdvice(input_0, Rotation::Cur());
      std::unique_ptr<Expression<F>> input_1_expr =
          meta.QueryFixed(input_1, Rotation::Cur());
      std::unique_ptr<Expression<F>> shuffle_0_expr =
          meta.QueryAdvice(shuffle_0, Rotation::Cur());
      std::unique_ptr<Expression<F>> shuffle_1_expr =
          meta.QueryAdvice(shuffle_1, Rotation::Cur());

      shuffle::Pairs<std::unique_ptr<Expression<F>>> shuffle_pairs;
      shuffle_pairs.emplace_back(
          s_input_expr->Clone() * std::move(input_0_expr),
          s_shuffle_expr->Clone() * std::move(shuffle_0_expr));
      shuffle_pairs.emplace_back(
          std::move(s_input_expr) * std::move(input_1_expr),
          std::move(s_shuffle_expr) * std::move(shuffle_1_expr));
      return shuffle_pairs;
    });

    return {input_0, input_1, shuffle_0, shuffle_1, s_input, s_shuffle};
  }

 private:
  AdviceColumnKey input_0_;
  FixedColumnKey input_1_;
  AdviceColumnKey shuffle_0_;
  AdviceColumnKey shuffle_1_;
  Selector s_input_;
  Selector s_shuffle_;
};

template <typename F, template <typename> class _FloorPlanner>
class ShuffleAPICircuit : public Circuit<ShuffleAPIConfig<F>> {
 public:
  using FloorPlanner = _FloorPlanner<ShuffleAPICircuit<F, _FloorPlanner>>;

  ShuffleAPICircuit() = default;
  ShuffleAPICircuit(std::vector<Value<F>>&& input_0, std::vector<F>&& input_1,
                    std::vector<Value<F>>&& shuffle_0,
                    std::vector<Value<F>>&& shuffle_1)
      : input_0_(std::move(input_0)),
        input_1_(std::move(input_1)),
        shuffle_0_(std::move(shuffle_0)),
        shuffle_1_(std::move(shuffle_1)) {
    CHECK_EQ(input_0.size(), input_1.size());
  }

  std::unique_ptr<Circuit<ShuffleAPIConfig<F>>> WithoutWitness()
      const override {
    return std::make_unique<ShuffleAPICircuit>();
  }

  static ShuffleAPIConfig<F> Configure(ConstraintSystem<F>& meta) {
    AdviceColumnKey input_0 = meta.CreateAdviceColumn();
    FixedColumnKey input_1 = meta.CreateFixedColumn();
    AdviceColumnKey shuffle_0 = meta.CreateAdviceColumn();
    AdviceColumnKey shuffle_1 = meta.CreateAdviceColumn();
    return ShuffleAPIConfig<F>::Configure(meta, input_0, input_1, shuffle_0,
                                          shuffle_1);
  }

  void Synthesize(ShuffleAPIConfig<F>&& config,
                  Layouter<F>* layouter) const override {
    layouter->AssignRegion("load inputs", [this, &config](Region<F>& region) {
      for (size_t i = 0; i < input_0_.size(); ++i) {
        region.AssignAdvice("input_0", config.input_0(), i,
                            [this, i]() { return input_0_[i]; });
        region.AssignFixed("input_1", config.input_1(), i, [this, i]() {
          return Value<F>::Known(input_1_[i]);
        });
        config.s_input().Enable(region, i);
      }
    });

    layouter->AssignRegion("load shuffles", [this, &config](Region<F>& region) {
      for (size_t i = 0; i < shuffle_0_.size(); ++i) {
        region.AssignAdvice("shuffle_0", config.shuffle_0(), i,
                            [this, i]() { return shuffle_0_[i]; });
        region.AssignAdvice("shuffle_1", config.shuffle_1(), i,
                            [this, i]() { return shuffle_1_[i]; });
        config.s_shuffle().Enable(region, i);
      }
    });
  }

 private:
  std::vector<Value<F>> input_0_;
  std::vector<F> input_1_;
  std::vector<Value<F>> shuffle_0_;
  std::vector<Value<F>> shuffle_1_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_API_CIRCUIT_H_
