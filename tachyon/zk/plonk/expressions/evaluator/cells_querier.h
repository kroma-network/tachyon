#ifndef TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_CELLS_QUERIER_H_
#define TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_CELLS_QUERIER_H_

#include "tachyon/base/logging.h"
#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"
#include "tachyon/zk/plonk/constraint_system/virtual_cells.h"
#include "tachyon/zk/plonk/expressions/advice_expression.h"
#include "tachyon/zk/plonk/expressions/fixed_expression.h"
#include "tachyon/zk/plonk/expressions/instance_expression.h"
#include "tachyon/zk/plonk/expressions/selector_expression.h"

namespace tachyon::zk::plonk {

template <typename F>
class CellsQuerier : public Evaluator<F, void> {
 public:
  explicit CellsQuerier(VirtualCells<F>& cells) : cells_(cells) {}

  // Evaluator methods
  void Evaluate(const Expression<F>* input) override {
    switch (input->type()) {
      case ExpressionType::kConstant:
        return;
      case ExpressionType::kNegated:
        Evaluate(input->ToNegated()->expr());
        return;
      case ExpressionType::kSum:
        Evaluate(input->ToSum()->left());
        Evaluate(input->ToSum()->right());
        return;
      case ExpressionType::kProduct:
        Evaluate(input->ToProduct()->left());
        Evaluate(input->ToProduct()->right());
        return;
      case ExpressionType::kScaled:
        Evaluate(input->ToScaled()->expr());
        return;
      case ExpressionType::kFixed: {
        FixedQuery& query = const_cast<FixedQuery&>(input->ToFixed()->query());
        if (!query.HasIndex()) {
          cells_.queried_cells_.emplace_back(query.column(), query.rotation());
          query.SetIndex(
              cells_.meta_->QueryFixedIndex(query.column(), query.rotation()));
        }
        return;
      }
      case ExpressionType::kAdvice: {
        AdviceQuery& query =
            const_cast<AdviceQuery&>(input->ToAdvice()->query());
        if (!query.HasIndex()) {
          cells_.queried_cells_.emplace_back(query.column(), query.rotation());
          query.SetIndex(
              cells_.meta_->QueryAdviceIndex(query.column(), query.rotation()));
        }
        return;
      }
      case ExpressionType::kInstance: {
        InstanceQuery& query =
            const_cast<InstanceQuery&>(input->ToInstance()->query());
        if (!query.HasIndex()) {
          cells_.queried_cells_.emplace_back(query.column(), query.rotation());
          query.SetIndex(cells_.meta_->QueryInstanceIndex(query.column(),
                                                          query.rotation()));
        }
        return;
      }
      case ExpressionType::kChallenge:
        return;
      case ExpressionType::kSelector: {
        Selector selector = input->ToSelector()->selector();
        // TODO(chokobole): should it be std::set<Selector>?
        if (!base::Contains(cells_.queried_selectors_, selector)) {
          cells_.queried_selectors_.push_back(selector);
        }
        return;
      }
      case ExpressionType::kFirstRow:
      case ExpressionType::kLastRow:
      case ExpressionType::kTransition:
      case ExpressionType::kVariable:
        NOTREACHED() << "AIR expression "
                     << ExpressionTypeToString(input->type())
                     << " is not allowed in plonk!";
    }
    NOTREACHED();
  }

 private:
  VirtualCells<F>& cells_;
};

template <typename F>
void QueryCells(const Expression<F>* input, VirtualCells<F>& cells) {
  CellsQuerier<F> querier(cells);
  querier.Evaluate(input);
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_CELLS_QUERIER_H_
