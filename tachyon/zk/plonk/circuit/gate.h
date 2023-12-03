#ifndef TACHYON_ZK_PLONK_CIRCUIT_GATE_H_
#define TACHYON_ZK_PLONK_CIRCUIT_GATE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/plonk/circuit/selector.h"
#include "tachyon/zk/plonk/circuit/virtual_cell.h"

namespace tachyon::zk {

template <typename F>
class Gate {
 public:
  Gate() = default;
  Gate(std::string name, std::vector<std::string> constraint_names,
       std::vector<std::unique_ptr<Expression<F>>> polys,
       std::vector<Selector> queried_selectors,
       std::vector<VirtualCell> queried_cells)
      : name_(std::move(name)),
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
  const std::vector<Selector>& queried_selectors() const {
    return queried_selectors_;
  }
  const std::vector<VirtualCell>& queried_cells() const {
    return queried_cells_;
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

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_GATE_H_
