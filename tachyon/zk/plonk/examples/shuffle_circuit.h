#ifndef TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_CIRCUIT_H_
#define TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_CIRCUIT_H_

#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/zk/base/row_index.h"
#include "tachyon/zk/plonk/constraint_system/circuit.h"

namespace tachyon::zk::plonk {

template <typename F, size_t W>
class ShuffleCircuitConfig {
 public:
  using Field = F;

  ShuffleCircuitConfig() = default;
  ShuffleCircuitConfig(Selector q_shuffle, Selector q_first, Selector q_last,
                       std::vector<AdviceColumnKey>&& original_column_keys,
                       std::vector<AdviceColumnKey>&& shuffled_column_keys,
                       Challenge theta, Challenge gamma, AdviceColumnKey z)
      : q_shuffle_(q_shuffle),
        q_first_(q_first),
        q_last_(q_last),
        original_column_keys_(std::move(original_column_keys)),
        shuffled_column_keys_(std::move(shuffled_column_keys)),
        theta_(theta),
        gamma_(gamma),
        z_(z) {
    CHECK_EQ(original_column_keys_.size(), W);
    CHECK_EQ(shuffled_column_keys_.size(), W);
  }

  ShuffleCircuitConfig Clone() const {
    std::vector<AdviceColumnKey> original_column_keys = original_column_keys_;
    std::vector<AdviceColumnKey> shuffled_column_keys = shuffled_column_keys_;
    return {q_shuffle_,
            q_first_,
            q_last_,
            std::move(original_column_keys),
            std::move(shuffled_column_keys),
            theta_,
            gamma_,
            z_};
  }

  Selector q_shuffle() const { return q_shuffle_; }
  Selector q_first() const { return q_first_; }
  Selector q_last() const { return q_last_; }
  const std::vector<AdviceColumnKey>& original_column_keys() const {
    return original_column_keys_;
  }
  const std::vector<AdviceColumnKey>& shuffled_column_keys() const {
    return shuffled_column_keys_;
  }
  Challenge theta() const { return theta_; }
  Challenge gamma() const { return gamma_; }
  const AdviceColumnKey& z() const { return z_; }

  static ShuffleCircuitConfig Configure(ConstraintSystem<F>& meta) {
    Selector q_shuffle = meta.CreateSimpleSelector();
    Selector q_first = meta.CreateSimpleSelector();
    Selector q_last = meta.CreateSimpleSelector();
    std::vector<AdviceColumnKey> original =
        base::CreateVector(W, [&meta]() { return meta.CreateAdviceColumn(); });
    std::vector<AdviceColumnKey> shuffled =
        base::CreateVector(W, [&meta]() { return meta.CreateAdviceColumn(); });
    Challenge theta = meta.CreateChallengeUsableAfter(kFirstPhase);
    Challenge gamma = meta.CreateChallengeUsableAfter(kFirstPhase);
    AdviceColumnKey z = meta.CreateAdviceColumn(kSecondPhase);

    meta.CreateGate("z should start with 1", [q_first,
                                              &z](VirtualCells<F>& meta) {
      std::unique_ptr<Expression<F>> q_first_expr = meta.QuerySelector(q_first);
      std::unique_ptr<Expression<F>> z_expr =
          meta.QueryAdvice(z, Rotation::Cur());
      std::unique_ptr<Expression<F>> one_expr =
          ExpressionFactory<F>::Constant(F::One());

      std::vector<Constraint<F>> constraints;
      constraints.emplace_back(std::move(q_first_expr) *
                               (std::move(one_expr) - std::move(z_expr)));
      return constraints;
    });

    meta.CreateGate("z should end with 1", [q_last, &z](VirtualCells<F>& meta) {
      std::unique_ptr<Expression<F>> q_last_expr = meta.QuerySelector(q_last);
      std::unique_ptr<Expression<F>> z_expr =
          meta.QueryAdvice(z, Rotation::Cur());
      std::unique_ptr<Expression<F>> one_expr =
          ExpressionFactory<F>::Constant(F::One());

      std::vector<Constraint<F>> constraints;
      constraints.emplace_back(std::move(q_last_expr) *
                               (std::move(one_expr) - std::move(z_expr)));
      return constraints;
    });

    meta.CreateGate(
        "z should have valid transition",
        [q_shuffle, &original, &shuffled, theta, gamma,
         &z](VirtualCells<F>& meta) {
          std::unique_ptr<Expression<F>> q_shuffle_expr =
              meta.QuerySelector(q_shuffle);

          std::vector<std::unique_ptr<Expression<F>>> original_exprs =
              base::CreateVector(W, [&meta, &original](size_t i) {
                return meta.QueryAdvice(original[i], Rotation::Cur());
              });

          std::vector<std::unique_ptr<Expression<F>>> shuffle_exprs =
              base::CreateVector(W, [&meta, &shuffled](size_t i) {
                return meta.QueryAdvice(shuffled[i], Rotation::Cur());
              });

          std::unique_ptr<Expression<F>> theta_expr =
              meta.QueryChallenge(theta);
          std::unique_ptr<Expression<F>> gamma_expr =
              meta.QueryChallenge(gamma);

          std::unique_ptr<Expression<F>> z_expr =
              meta.QueryAdvice(z, Rotation::Cur());
          std::unique_ptr<Expression<F>> z_w_expr =
              meta.QueryAdvice(z, Rotation::Next());

          // Compress
          std::unique_ptr<Expression<F>> original_acc_expr = std::accumulate(
              original_exprs.begin() + 1, original_exprs.end(),
              std::move(original_exprs[0]),
              [&theta_expr](std::unique_ptr<Expression<F>>& acc,
                            std::unique_ptr<Expression<F>>& expr) {
                return std::move(acc) * theta_expr + std::move(expr);
              });

          // Compress
          std::unique_ptr<Expression<F>> shuffled_acc_expr = std::accumulate(
              shuffle_exprs.begin() + 1, shuffle_exprs.end(),
              std::move(shuffle_exprs[0]),
              [&theta_expr](std::unique_ptr<Expression<F>>& acc,
                            const std::unique_ptr<Expression<F>>& expr) {
                return std::move(acc) * theta_expr + std::move(expr);
              });

          // TODO(dongchangYoo): Only one instance of |gamma_expr| should be
          // moved to avoid an unnecessary deep copy. However, the order of
          // evaluating expression varies by compiler so applying |std::move| to
          // only one of the |gamma_expr|s may cause an error. Therefore, the
          // following code has been implemented in a way that copies both.
          // In the future, we plan to change |gamma_expr| type to |shared_ptr|
          // to enable shallow copying.
          std::vector<Constraint<F>> constraints;
          constraints.emplace_back(
              std::move(q_shuffle_expr) *
              (std::move(z_expr) * (std::move(original_acc_expr) + gamma_expr) -
               std::move(z_w_expr) *
                   (std::move(shuffled_acc_expr) + gamma_expr)));
          return constraints;
        });

    return {q_shuffle,           q_first, q_last, std::move(original),
            std::move(shuffled), theta,   gamma,  z};
  }

 private:
  Selector q_shuffle_;
  Selector q_first_;
  Selector q_last_;
  std::vector<AdviceColumnKey> original_column_keys_;
  std::vector<AdviceColumnKey> shuffled_column_keys_;
  Challenge theta_;
  Challenge gamma_;
  AdviceColumnKey z_;
};

template <typename F, size_t W, RowIndex H,
          template <typename> class _FloorPlanner>
class ShuffleCircuit : public Circuit<ShuffleCircuitConfig<F, W>> {
 public:
  using FloorPlanner = _FloorPlanner<ShuffleCircuit<F, W, H, _FloorPlanner>>;

  ShuffleCircuit() = default;
  ShuffleCircuit(std::vector<std::vector<F>>&& original_table,
                 std::vector<std::vector<F>>&& shuffled_table)
      : original_table_(std::move(original_table)),
        shuffled_table_(std::move(shuffled_table)) {
    CHECK_EQ(original_table_.size(), W);
    CHECK_EQ(shuffled_table_.size(), W);
    for (size_t i = 0; i < W; ++i) {
      CHECK_EQ(original_table_[i].size(), H);
      CHECK_EQ(shuffled_table_[i].size(), H);
    }
  }

  std::unique_ptr<Circuit<ShuffleCircuitConfig<F, W>>> WithoutWitness()
      const override {
    std::vector<std::vector<F>> dummy_original_table = base::CreateVector(
        W, []() { return base::CreateVector(H, F::Zero()); });
    std::vector<std::vector<F>> dummy_shuffled_table = base::CreateVector(
        W, []() { return base::CreateVector(H, F::Zero()); });
    ShuffleCircuit dummy_circuit(std::move(dummy_original_table),
                                 std::move(dummy_shuffled_table));
    return std::make_unique<ShuffleCircuit>(std::move(dummy_circuit));
  }

  static ShuffleCircuitConfig<F, W> Configure(ConstraintSystem<F>& meta) {
    return ShuffleCircuitConfig<F, W>::Configure(meta);
  }

  void Synthesize(ShuffleCircuitConfig<F, W>&& config,
                  Layouter<F>* layouter) const override {
    Value<F> theta = layouter->GetChallenge(config.theta());
    Value<F> gamma = layouter->GetChallenge(config.gamma());

    layouter->AssignRegion(
        "Shuffle original into shuffled",
        [this, &config, &theta, &gamma](Region<F>& region) {
          // Keygen
          config.q_first().Enable(region, 0);
          config.q_last().Enable(region, H);
          for (RowIndex i = 0; i < H; ++i) {
            config.q_shuffle().Enable(region, i);
          }

          // First phase
          for (size_t i = 0; i < W; ++i) {
            const AdviceColumnKey& original_column_key =
                config.original_column_keys()[i];
            const std::vector<F>& original_values = original_table_[i];
            for (RowIndex j = 0; j < H; ++j) {
              region.AssignAdvice(absl::Substitute("original[$0][$1]", i, j),
                                  original_column_key, j,
                                  [&original_values, j]() {
                                    return Value<F>::Known(original_values[j]);
                                  });
            }
          }
          for (size_t i = 0; i < W; ++i) {
            const AdviceColumnKey& shuffled_column_key =
                config.shuffled_column_keys()[i];
            const std::vector<F>& shuffled_values = shuffled_table_[i];
            for (RowIndex j = 0; j < H; ++j) {
              region.AssignAdvice(absl::Substitute("shuffled[$0][$1]", i, j),
                                  shuffled_column_key, j,
                                  [&shuffled_values, j]() {
                                    return Value<F>::Known(shuffled_values[j]);
                                  });
            }
          }

          // Second phase
          std::vector<Value<F>> z;
          z.reserve(H + 1);
          if (!theta.IsNone() && !gamma.IsNone()) {
            std::vector<F> product =
                base::CreateVector(H, [this, &theta, &gamma](RowIndex i) {
                  F compressed = std::accumulate(
                      shuffled_table_.begin(), shuffled_table_.end(), F::Zero(),
                      [&theta, i](F& acc,
                                  const std::vector<F>& shuffled_column) {
                        acc *= theta.value();
                        return acc += shuffled_column[i];
                      });
                  compressed += gamma.value();
                  return compressed;
                });

            F::BatchInverseInPlace(product);

            for (RowIndex i = 0; i < H; ++i) {
              F compressed = std::accumulate(
                  original_table_.begin(), original_table_.end(), F::Zero(),
                  [&theta, i](F& acc, const std::vector<F>& original_column) {
                    acc *= theta.value();
                    return acc += original_column[i];
                  });
              compressed += gamma.value();
              product[i] *= compressed;
            }

            z[0] = Value<F>::Known(F::One());
            for (RowIndex i = 0; i < H; ++i) {
              z[i + 1] = z[i] * Value<F>::Known(product[i]);
            }
          } else {
            z = base::CreateVector(H + 1, Value<F>::Unknown());
          }

          for (RowIndex i = 0; i < H + 1; ++i) {
            region.AssignAdvice(absl::Substitute("z[$0]", i), config.z(), i,
                                [&z, i]() { return z[i]; });
          }
        });
  }

 private:
  std::vector<std::vector<F>> original_table_;
  std::vector<std::vector<F>> shuffled_table_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_CIRCUIT_H_
