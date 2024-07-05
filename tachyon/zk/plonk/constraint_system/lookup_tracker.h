#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_LOOKUP_TRACKER_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_LOOKUP_TRACKER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk::plonk {

template <typename F>
struct LookupTracker {
  std::string name;
  std::vector<std::unique_ptr<Expression<F>>> table;
  std::vector<std::vector<std::unique_ptr<Expression<F>>>> inputs;
  LookupTracker() = default;
  LookupTracker(std::string_view name,
                std::vector<std::unique_ptr<Expression<F>>> table,
                std::vector<std::unique_ptr<Expression<F>>> input)
      : name(std::string(name)), table(std::move(table)) {
    inputs.push_back(std::move(input));
  }
  LookupTracker(std::string_view name,
                std::vector<std::unique_ptr<Expression<F>>> table,
                std::vector<std::vector<std::unique_ptr<Expression<F>>>> input)
      : name(std::string(name)),
        table(std::move(table)),
        inputs(std::move(input)) {}
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_LOOKUP_TRACKER_H_
