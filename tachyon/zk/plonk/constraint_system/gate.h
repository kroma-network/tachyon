#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_GATE_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_GATE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/plonk/constraint_system/selector.h"
#include "tachyon/zk/plonk/constraint_system/virtual_cell.h"

namespace tachyon::zk::plonk {

template <typename F>
class Gate {
 public:
  Gate() = default;
  Gate(std::string_view name, std::vector<std::string>&& constraint_names,
       std::vector<std::unique_ptr<Expression<F>>>&& polys,
       std::vector<Selector>&& queried_selectors,
       std::vector<VirtualCell>&& queried_cells)
      : name_(std::string(name)),
        constraint_names_(std::move(constraint_names)),
        polys_(std::move(polys)),
        queried_selectors_(std::move(queried_selectors)),
        queried_cells_(std::move(queried_cells)) {}

  const std::string& name() const { return name_; }
  const std::vector<std::string>& constraint_names() const {
    return constraint_names_;
  }
  const std::vector<std::unique_ptr<Expression<F>>>& polys() const {
    return polys_;
  }
  std::vector<std::unique_ptr<Expression<F>>>& polys() { return polys_; }
  const std::vector<Selector>& queried_selectors() const {
    return queried_selectors_;
  }
  const std::vector<VirtualCell>& queried_cells() const {
    return queried_cells_;
  }

  bool operator==(const Gate& other) const {
    if (!(name_ == other.name_ &&
          constraint_names_ == other.constraint_names_ &&
          queried_selectors_ == other.queried_selectors_ &&
          queried_cells_ == other.queried_cells_))
      return false;
    if (polys_.size() != other.polys_.size()) return false;
    for (size_t i = 0; i < polys_.size(); ++i) {
      if (*polys_[i] != *other.polys_[i]) return false;
    }
    return true;
  }
  bool operator!=(const Gate& other) const { return !operator==(other); }

  std::string ToString() const {
    std::vector<std::string> polys_str =
        base::Map(polys_, [](const std::unique_ptr<Expression<F>>& expr) {
          return expr->ToString();
        });
    std::vector<std::string> queried_selectors_str =
        base::Map(queried_selectors_,
                  [](const Selector& selector) { return selector.ToString(); });
    std::vector<std::string> queried_cells_str =
        base::Map(queried_cells_, [](const VirtualCell& queried_cell) {
          return queried_cell.ToString();
        });
    return absl::Substitute(
        "name: $0, constraint_names: [$1], polys: [$2], queried_selectors: "
        "[$3], queried_cells: [$4]",
        name_, absl::StrJoin(constraint_names_, ", "),
        absl::StrJoin(polys_str, ", "),
        absl::StrJoin(queried_selectors_str, ", "),
        absl::StrJoin(queried_cells_str, ", "));
  }

 private:
  std::string name_;
  std::vector<std::string> constraint_names_;
  std::vector<std::unique_ptr<Expression<F>>> polys_;
  // We track queried selectors separately from other cells, so that we can use
  // them to trigger debug checks on gates.
  std::vector<Selector> queried_selectors_;
  std::vector<VirtualCell> queried_cells_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_GATE_H_
