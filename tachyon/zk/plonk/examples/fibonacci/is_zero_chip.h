#ifndef TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_IS_ZERO_CHIP_H_
#define TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_IS_ZERO_CHIP_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/layout/region.h"

namespace tachyon::zk::plonk {

template <typename F>
class IsZeroConfig {
 public:
  using Field = F;

  IsZeroConfig(const AdviceColumnKey& value_inv,
               std::unique_ptr<Expression<F>> is_zero_expr)
      : value_inv_(value_inv), is_zero_expr_(std::move(is_zero_expr)) {}

  const AdviceColumnKey& value_inv() const { return value_inv_; }
  std::unique_ptr<Expression<F>> expr() const { return is_zero_expr_->Clone(); }

  IsZeroConfig Clone() const { return {value_inv_, is_zero_expr_->Clone()}; }

 private:
  AdviceColumnKey value_inv_;
  std::unique_ptr<Expression<F>> is_zero_expr_;
};

template <typename F>
class IsZeroChip {
 public:
  explicit IsZeroChip(IsZeroConfig<F>&& config) : config_(std::move(config)) {}

  static IsZeroConfig<F> Configure(
      ConstraintSystem<F>& meta,
      base::OnceCallback<std::unique_ptr<Expression<F>>(VirtualCells<F>&)>
          q_enable,
      base::OnceCallback<std::unique_ptr<Expression<F>>(VirtualCells<F>&)>
          value,
      const AdviceColumnKey& value_inv) {
    std::unique_ptr<Expression<F>> is_zero_expr =
        ExpressionFactory<F>::Constant(F::Zero());

    meta.CreateGate("is_zero", [&q_enable, &value, value_inv,
                                &is_zero_expr](VirtualCells<F>& meta) mutable {
      // clang-format off
      //
      // q_enable | value |  value_inv |  1 - value * value_inv | value * (1 - value * value_inv)
      // ---------+-------+------------+------------------------+-------------------------------
      //   yes    |   x   |    1/x     |         0              |         0
      //   no     |   x   |    0       |         1              |         x
      //   yes    |   0   |    0       |         1              |         0
      //   yes    |   0   |    y       |         1              |         0
      //
      // clang-format on
      std::unique_ptr<Expression<F>> q_enable_res =
          std::move(q_enable).Run(meta);
      std::unique_ptr<Expression<F>> value_res = std::move(value).Run(meta);
      std::unique_ptr<Expression<F>> value_inv_res =
          meta.QueryAdvice(value_inv, Rotation::Cur());

      is_zero_expr = ExpressionFactory<F>::Constant(F::One()) -
                     value_res->Clone() * std::move(value_inv_res);
      std::vector<Constraint<F>> constraints;
      constraints.emplace_back(std::move(q_enable_res) * std::move(value_res) *
                               is_zero_expr->Clone());
      return constraints;
    });

    return IsZeroConfig<F>(value_inv, std::move(is_zero_expr));
  }

  void Assign(Region<F>& region, RowIndex offset, const Value<F>& value) const {
    const F value_inv = value.IsZero() ? F::Zero() : value.value().Inverse();
    region.AssignAdvice("value inv", config_.value_inv(), offset,
                        [&value_inv]() { return Value<F>::Known(value_inv); });
  }

 private:
  IsZeroConfig<F> config_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_IS_ZERO_CHIP_H_
