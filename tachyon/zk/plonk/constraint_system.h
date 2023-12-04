// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_H_

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/containers/contains.h"
#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/expressions/evaluator/simple_selector_finder.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/plonk/circuit/constraint.h"
#include "tachyon/zk/plonk/circuit/gate.h"
#include "tachyon/zk/plonk/circuit/lookup_table_column.h"
#include "tachyon/zk/plonk/circuit/query.h"
#include "tachyon/zk/plonk/circuit/selector_compressor.h"
#include "tachyon/zk/plonk/circuit/virtual_cells.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"

namespace tachyon::zk {

// This is a description of the circuit environment, such as the gate, column
// and permutation arrangements.
template <typename F>
class ConstraintSystem {
 public:
  using LookupCallback =
      base::OnceCallback<LookupPairs<std::unique_ptr<Expression<F>>,
                                     LookupTableColumn>(VirtualCells<F>&)>;
  using LookupAnyCallback =
      base::OnceCallback<LookupPairs<std::unique_ptr<Expression<F>>>(
          VirtualCells<F>&)>;
  using ConstrainCallback =
      base::OnceCallback<std::vector<Constraint<F>>(VirtualCells<F>&)>;

  size_t num_fixed_columns() const { return num_fixed_columns_; }

  size_t num_advice_columns() const { return num_advice_columns_; }

  size_t num_instance_columns() const { return num_instance_columns_; }

  size_t num_selectors() const { return num_selectors_; }

  size_t num_challenges() const { return num_challenges_; }

  const std::vector<Phase>& advice_column_phases() const {
    return advice_column_phases_;
  }

  const std::vector<Phase>& challenge_phases() const {
    return challenge_phases_;
  }

  const std::vector<FixedColumnKey>& selector_map() const {
    return selector_map_;
  }

  const std::vector<Gate<F>>& gates() const { return gates_; }

  const std::vector<AdviceQueryData>& advice_queries() const {
    return advice_queries_;
  }

  const std::vector<size_t>& num_advice_queries() const {
    return num_advice_queries_;
  }

  const std::vector<InstanceQueryData>& instance_queries() const {
    return instance_queries_;
  }

  const std::vector<FixedQueryData>& fixed_queries() const {
    return fixed_queries_;
  }

  const PermutationArgument& permutation() const { return permutation_; }

  const std::vector<LookupArgument<F>>& lookups() const { return lookups_; }

  const absl::flat_hash_map<ColumnKeyBase, std::string>&
  general_column_annotations() const {
    return general_column_annotations_;
  }

  const std::vector<FixedColumnKey>& constants() const { return constants_; }

  const std::optional<size_t>& minimum_degree() const {
    return minimum_degree_;
  }

  // Sets the minimum degree required by the circuit, which can be set to a
  // larger amount than actually needed. This can be used, for example, to
  // force the permutation argument to involve more columns in the same set.
  void set_minimum_degree(size_t degree) { minimum_degree_ = degree; }

  // Enables this fixed |column| to be used for global constant assignments.
  // The |column| will be equality-enabled, too.
  void EnableConstant(const FixedColumnKey& column) {
    // TODO(chokobole): should it be std::set<FixedColumnKey>?
    if (!base::Contains(constants_, column)) {
      constants_.push_back(column);
      EnableEquality(AnyColumnKey(column));
    }
  }

  // Enable the ability to enforce equality over cells in this column
  void EnableEquality(const AnyColumnKey& column) {
    // TODO(chokobole): should it be std::set<FixedColumnKey>?
    constants_.push_back(column);
    EnableEquality(column);
    permutation_.AddColumn(column);
  }

  // Add a lookup argument for some input expressions and table columns.
  //
  // |callback| returns a map between the input expressions and the table
  // columns they need to match.
  size_t Lookup(std::string_view name, LookupCallback callback) {
    VirtualCells cells(this);
    LookupPairs<std::unique_ptr<Expression<F>>> pairs = base::Map(
        std::move(callback).Run(cells),
        [&cells](LookupPair<std::unique_ptr<Expression<F>>, LookupTableColumn>&
                     pair) {
          CHECK(!pair.input()->ContainsSimpleSelector())
              << "expression containing simple selector "
                 "supplied to lookup argument";

          std::unique_ptr<Expression<F>> table =
              cells.QueryFixed(pair.table().column(), Rotation::Cur());

          return LookupPair<std::unique_ptr<Expression<F>>>(
              std::move(pair).input(), std::move(table));
        });

    lookups_.emplace_back(name, std::move(pairs));
    return lookups_.size() - 1;
  }

  // Add a lookup argument for some input expressions and table expressions.
  //
  // |callback| returns a map between the input expressions and the table
  // expressions they need to match.
  size_t LookupAny(std::string_view name, LookupAnyCallback callback) {
    VirtualCells cells(this);
    LookupPairs<std::unique_ptr<Expression<F>>> pairs =
        std::move(callback).Run(cells);

    lookups_.emplace_back(name, std::move(pairs));
    return lookups_.size() - 1;
  }

  size_t QueryFixedIndex(const FixedColumnKey& column, Rotation at) {
    // Return existing query, if it exists
    size_t index;
    if (QueryIndex(fixed_queries_, column, at, &index)) return index;

    // Make a new query
    fixed_queries_.emplace_back(at, column);
    return fixed_queries_.size() - 1;
  }

  size_t QueryAdviceIndex(const AdviceColumnKey& column, Rotation at) {
    // Return existing query, if it exists
    size_t index;
    if (QueryIndex(advice_queries_, column, at, &index)) return index;

    // Make a new query
    advice_queries_.emplace_back(at, column);
    num_advice_queries_[column.index()] += 1;
    return advice_queries_.size() - 1;
  }

  size_t QueryInstanceIndex(const InstanceColumnKey& column, Rotation at) {
    // Return existing query, if it exists
    size_t index;
    if (QueryIndex(instance_queries_, column, at, &index)) return index;

    // Make a new query
    instance_queries_.emplace_back(at, column);
    return instance_queries_.size() - 1;
  }

  size_t QueryAnyIndex(const AnyColumnKey& column, Rotation at) {
    switch (column.type()) {
      case ColumnType::kFixed:
        return QueryFixedIndex(FixedColumnKey(column), at);
      case ColumnType::kAdvice:
        return QueryAdviceIndex(AdviceColumnKey(column), at);
      case ColumnType::kInstance:
        return QueryInstanceIndex(InstanceColumnKey(column), at);
      case ColumnType::kAny:
        break;
    }
    NOTREACHED();
    return 0;
  }

  size_t GetFixedQueryIndex(const FixedColumnKey& column, Rotation at) const {
    size_t index;
    CHECK(QueryIndex(fixed_queries_, column, at, &index));
    return index;
  }

  size_t GetAdviceQueryIndex(const AdviceColumnKey& column, Rotation at) const {
    size_t index;
    CHECK(QueryIndex(advice_queries_, column, at, &index));
    return index;
  }

  size_t GetInstanceQueryIndex(const InstanceColumnKey& column,
                               Rotation at) const {
    size_t index;
    CHECK(QueryIndex(instance_queries_, column, at, &index));
    return index;
  }

  size_t GetAnyQueryIndex(const AnyColumnKey& column, Rotation at) const {
    switch (column.type()) {
      case ColumnType::kFixed:
        return GetFixedQueryIndex(FixedColumnKey(column), at);
      case ColumnType::kAdvice:
        return GetAdviceQueryIndex(AdviceColumnKey(column), at);
      case ColumnType::kInstance:
        return GetInstanceQueryIndex(InstanceColumnKey(column), at);
      case ColumnType::kAny:
        break;
    }
    NOTREACHED();
    return 0;
  }

  // Creates a new gate.
  //
  // A gate is required to contain polynomial constraints. This method will
  // panic if |constrain| returns an empty iterator.
  void CreateGate(std::string_view name, ConstrainCallback constrain) {
    VirtualCells cells(this);
    std::vector<Constraint<F>> constraints = std::move(constraints).Run(cells);
    std::vector<Selector> queried_selectors =
        std::move(cells).queried_selectors();
    std::vector<VirtualCell> queried_cells = std::move(cells).queried_cells();

    std::vector<std::string> constraint_names;
    std::vector<std::unique_ptr<Expression<F>>> polys;
    for (Constraint<F>& constraint : constraints) {
      constraint_names.push_back(std::move(constraint).name());
      polys.push_back(std::move(constraint).expression());
    }
    CHECK(!polys.empty()) << "Gates must contain at least one constraint.";

    gates_.emplace_back(name, std::move(constraint_names), std::move(polys),
                        std::move(queried_selectors), std::move(queried_cells));
  }

  // This will compress selectors together depending on their provided
  // assignments. This |ConstraintSystem| will then be modified to add new
  // fixed columns (representing the actual selectors) and will return the
  // polynomials for those columns. Finally, an internal map is updated to
  // find which fixed column corresponds with a given |Selector|.
  //
  // Do not call this twice. Yes, this should be a builder pattern instead.
  std::vector<std::vector<F>> CompressSelectors(
      const std::vector<std::vector<bool>>& selectors) {
    // The number of provided selector assignments must be the number we
    // counted for this constraint system.
    CHECK_EQ(selectors.size(), num_selectors_);

    // Compute the maximal degree of every selector. We only consider the
    // expressions in gates, as lookup arguments cannot support simple
    // selectors. Selectors that are complex or do not appear in any gates
    // will have degree zero.
    std::vector<size_t> degrees =
        base::CreateVector(selectors.size(), size_t{0});
    for (const Gate<F>& gate : gates_) {
      for (const std::unique_ptr<Expression<F>>& expression : gate.polys()) {
        std::optional<Selector> selector = expression->ExtractSimpleSelector();
        if (selector.has_value()) {
          degrees[selector->index()] =
              std::max(degrees[selector->index()], expression->Degree());
        }
      }
    }

    // We will not increase the degree of the constraint system, so we limit
    // ourselves to the largest existing degree constraint.
    std::vector<FixedColumnKey> new_columns;
    typename SelectorCompressor<F>::Result result =
        SelectorCompressor<F>::Process(
            std::move(selectors), degrees, ComputeDegree(),
            [this, &new_columns]() {
              FixedColumnKey column = CreateFixedColumn();
              new_columns.push_back(column);
              return ExpressionFactory<F>::Fixed(
                  FixedQuery(QueryFixedIndex(column, Rotation::Cur()),
                             column.index(), Rotation::Cur()));
            });

    std::vector<FixedColumnKey> selector_map;
    selector_map.resize(result.assignments.size());
    std::vector<Ref<const Expression<F>>> selector_replacements =
        base::CreateVector(result.assignments.size(),
                           Ref<const Expression<F>>());
    for (const SelectorAssignment<F>& assignment : result.assignments) {
      selector_replacements[assignment.selector_index()] =
          Ref<const Expression<F>>(assignment.expression());
      selector_map[assignment.selector_index()] =
          new_columns[assignment.combination_index]();
    }

    selector_map_ = std::move(selector_map);

    for (Gate<F>& gate : gates_) {
      for (std::unique_ptr<Expression<F>>& expression : gate.polys()) {
        expression = expression->ReplaceSelectors(selector_replacements, false);
      }
    }
    for (LookupArgument<F>& lookup : lookups_) {
      for (std::unique_ptr<Expression<F>>& expression :
           lookup.input_expressions()) {
        expression = expression->ReplaceSelectors(selector_replacements, true);
      }
      for (std::unique_ptr<Expression<F>>& expression :
           lookup.table_expressions()) {
        expression = expression->ReplaceSelectors(selector_replacements, true);
      }
    }

    return result.polys;
  }

  // Allocate a new simple selector. Simple selectors cannot be added to
  // expressions nor multiplied by other expressions containing simple
  // selectors. Also, simple selectors may not appear in lookup argument
  // inputs.
  Selector CreateSimpleSelector() { return Selector::Simple(num_selectors_++); }

  // Allocate a new complex selector that can appear anywhere
  // within expressions.
  Selector CreateComplexSelector() {
    return Selector::Complex(num_selectors_++);
  }

  // Allocate a new fixed column that can be used in a lookup table.
  LookupTableColumn CreateLookupTableColumn() {
    return LookupTableColumn(CreateFixedColumn());
  }

  // Annotate a Lookup column.
  void AnnotateLookupColumn(const LookupTableColumn& column,
                            std::string_view name) {
    // We don't care if the table has already an annotation. If it's the case we
    // keep the new one.
    general_column_annotations_[ColumnKeyBase(
        ColumnType::kFixed, column.column().index())] = std::string(name);
  }

  // Annotate an Any column.
  void AnnotateLookupAnyColumn(const AnyColumnKey& column,
                               std::string_view name) {
    // We don't care if the table has already an annotation. If it's the case we
    // keep the new one.
    general_column_annotations_[ColumnKeyBase(column.type(), column.index())] =
        std::string(name);
  }

  // Allocate a new fixed column
  FixedColumnKey CreateFixedColumn() {
    return FixedColumnKey(num_fixed_columns_++);
  }

  // Allocate a new advice column at |kFirstPhase|.
  AdviceColumnKey CreateAdviceColumn() {
    return CreateAdviceColumn(kFirstPhase);
  }

  // Allocate a new advice column in given phase
  AdviceColumnKey CreateAdviceColumn(Phase phase) {
    Phase previous_phase;
    if (phase.Prev(&previous_phase)) {
      CHECK(base::Contains(advice_column_phases_, previous_phase))
          << "Phase " << previous_phase.ToString() << "is not used";
    }

    AdviceColumnKey column(num_advice_columns_++, phase);
    num_advice_queries_.push_back(0);
    advice_column_phases_.push_back(phase);
    return column;
  }

  // Allocate a new instance column
  InstanceColumnKey CreateInstanceColumn() {
    return InstanceColumnKey(num_instance_columns_++);
  }

  Challenge CreateChallengeUsableAfter(Phase phase) {
    CHECK(base::Contains(advice_column_phases_, phase))
        << "Phase " << phase.ToString() << "is not used";
    Challenge challenge(num_challenges_++, phase);
    challenge_phases_.push_back(phase);
    return challenge;
  }

  Phase ComputeMaxPhase() const {
    auto max_phase_it = std::max_element(advice_column_phases_.begin(),
                                         advice_column_phases_.end());
    if (max_phase_it == advice_column_phases_.end()) return Phase(0);
    return *max_phase_it;
  }

  std::vector<Phase> GetPhases() const {
    Phase max_phase = ComputeMaxPhase();
    return base::CreateVector(size_t{max_phase.value()}, [](size_t i) {
      return Phase(static_cast<uint8_t>(i));
    });
  }

  // Compute the degree of the constraint system (the maximum degree of all
  // constraints).
  size_t ComputeDegree() const {
    // The permutation argument will serve alongside the gates, so must be
    // accounted for.
    size_t degree = permutation_.RequiredDegree();

    // The lookup argument also serves alongside the gates and must be accounted
    // for.
    degree = std::max(degree, ComputeLookupRequiredDegree());

    // Account for each gate to ensure our quotient polynomial is the
    // correct degree and that our extended domain is the right size.
    degree = std::max(degree, ComputeGateRequiredDegree());

    return std::max(degree, minimum_degree_.value_or(1));
  }

  // Compute the number of blinding factors necessary to perfectly blind
  // each of the prover's witness polynomials.
  size_t ComputeBlindingFactors() const {
    // All of the prover's advice columns are evaluated at no more than
    auto max_num_advice_query_it = std::max_element(num_advice_queries_.begin(),
                                                    num_advice_queries_.end());
    size_t factors = max_num_advice_query_it == num_advice_queries_.end()
                         ? 1
                         : *max_num_advice_query_it;
    // distinct points during gate checks.

    // - The permutation argument witness polynomials are evaluated at most 3
    //   times.
    // - Each lookup argument has independent witness polynomials, and they are
    //   evaluated at most 2 times.
    factors = std::max(size_t{3}, factors);

    // Each polynomial is evaluated at most an additional time during
    // multiopen (at x₃ to produce qₑᵥₐₗₛ):
    ++factors;

    // h(x) is derived by the other evaluations so it does not reveal
    // anything; in fact it does not even appear in the proof.

    // h(x₃) is also not revealed; the verifier only learns a single
    // evaluation of a polynomial in x₁ which has h(x₃) and another random
    // polynomial evaluated at x₃ as coefficients -- this random polynomial
    // is "random_poly" in the vanishing argument.

    // Add an additional blinding factor as a slight defense against
    // off-by-one errors.
    return ++factors;
  }

  // Returns the minimum necessary rows that need to exist in order to
  // account for e.g. blinding factors.
  size_t ComputeMinimumRows() const {
    return ComputeBlindingFactors()  // m blinding factors
           + 1                       // for l_{-(m + 1)} (lₗₐₛₜ)
           + 1  // for l₀ (just for extra breathing room for the permutation
                // argument, to essentially force a separation in the
           // permutation polynomial between the roles of lₗₐₛₜ, l₀
           // and the interstitial values.)
           + 1;  // for at least one row
  }

 private:
  template <typename QueryDataTy, typename ColumnTy>
  static bool QueryIndex(const std::vector<QueryDataTy>& queries,
                         const ColumnTy& column, Rotation at,
                         size_t* index_out) {
    std::optional<size_t> index =
        base::FindIndexIf(queries, [&column, at](const QueryDataTy& query) {
          return query.column() == column && query.rotation() == at;
        });
    if (!index.has_value()) return false;
    *index = index.value();
    return true;
  }

  size_t ComputeLookupRequiredDegree() const {
    std::vector<size_t> required_degrees =
        base::Map(lookups_, [](const LookupArgument<F>& argument) {
          return argument.RequiredDegree();
        });
    auto max_required_degree =
        std::max_element(required_degrees.begin(), required_degrees.end());
    if (max_required_degree == required_degrees.end()) return 1;
    return *max_required_degree;
  }

  size_t ComputeGateRequiredDegree() const {
    std::vector<size_t> required_degrees =
        base::FlatMap(gates_, [](const Gate<F>& gate) {
          return base::Map(gate.polys,
                           [](const std::unique_ptr<Expression<F>>& poly) {
                             return poly->Degree();
                           });
        });
    auto max_required_degree =
        std::max_element(required_degrees.begin(), required_degrees.end());
    if (max_required_degree == required_degrees.end()) return 0;
    return *max_required_degree;
  }

  size_t num_fixed_columns_ = 0;
  size_t num_advice_columns_ = 0;
  size_t num_instance_columns_ = 0;
  size_t num_selectors_ = 0;
  size_t num_challenges_ = 0;

  // Contains the phase for each advice column. Should have same length as
  // num_advice_columns.
  std::vector<Phase> advice_column_phases_;
  // Contains the phase for each challenge. Should have same length as
  // num_challenges.
  std::vector<Phase> challenge_phases_;

  // This is a cached vector that maps virtual selectors to the concrete
  // fixed column that they were compressed into. This is just used by dev
  // tooling right now.
  std::vector<FixedColumnKey> selector_map_;
  std::vector<Gate<F>> gates_;
  std::vector<AdviceQueryData> advice_queries_;
  // Contains an integer for each advice column
  // identifying how many distinct queries it has
  // so far; should be same length as num_advice_columns.
  std::vector<size_t> num_advice_queries_;
  std::vector<InstanceQueryData> instance_queries_;
  std::vector<FixedQueryData> fixed_queries_;

  // Permutation argument for performing equality constraints.
  PermutationArgument permutation_;

  // Vector of lookup arguments, where each corresponds
  // to a sequence of input expressions and a sequence
  // of table expressions involved in the lookup.
  std::vector<LookupArgument<F>> lookups_;

  // List of indexes of Fixed columns which are associated to a
  // circuit-general Column tied to their annotation.
  absl::flat_hash_map<ColumnKeyBase, std::string> general_column_annotations_;

  // Vector of fixed columns, which can be used to store constant values
  // that are copied into advice columns.
  std::vector<FixedColumnKey> constants_;

  std::optional<size_t> minimum_degree_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_H_
