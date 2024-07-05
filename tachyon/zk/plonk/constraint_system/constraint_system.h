// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/numeric/bits.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/containers/contains.h"
#include "tachyon/base/functional/callback.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/zk/base/row_types.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/lookup/type.h"
#include "tachyon/zk/plonk/constraint_system/constraint.h"
#include "tachyon/zk/plonk/constraint_system/gate.h"
#include "tachyon/zk/plonk/constraint_system/lookup_tracker.h"
#include "tachyon/zk/plonk/constraint_system/query.h"
#include "tachyon/zk/plonk/constraint_system/selector_compressor.h"
#include "tachyon/zk/plonk/constraint_system/virtual_cells.h"
#include "tachyon/zk/plonk/expressions/evaluator/identifier.h"
#include "tachyon/zk/plonk/expressions/evaluator/selector_replacer.h"
#include "tachyon/zk/plonk/expressions/evaluator/simple_selector_extractor.h"
#include "tachyon/zk/plonk/expressions/evaluator/simple_selector_finder.h"
#include "tachyon/zk/plonk/keys/c_proving_key_impl_forward.h"
#include "tachyon/zk/plonk/layout/lookup_table_column.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_utils.h"

namespace tachyon::zk::plonk {

// This is a description of the circuit environment, such as the gate, column
// and permutation arrangements.
template <typename F>
class ConstraintSystem {
 public:
  using LookupCallback =
      base::OnceCallback<lookup::Pairs<std::unique_ptr<Expression<F>>,
                                       LookupTableColumn>(VirtualCells<F>&)>;
  using LookupAnyCallback =
      base::OnceCallback<lookup::Pairs<std::unique_ptr<Expression<F>>>(
          VirtualCells<F>&)>;
  using ConstrainCallback =
      base::OnceCallback<std::vector<Constraint<F>>(VirtualCells<F>&)>;

  ConstraintSystem() = default;

  explicit ConstraintSystem(lookup::Type lookup_type)
      : lookup_type_(lookup_type) {}

  lookup::Type lookup_type() const { return lookup_type_; }

  size_t num_fixed_columns() const { return num_fixed_columns_; }

  size_t num_advice_columns() const { return num_advice_columns_; }

  size_t num_instance_columns() const { return num_instance_columns_; }

  size_t num_simple_selectors() const { return num_simple_selectors_; }

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

  const std::vector<RowIndex>& num_advice_queries() const {
    return num_advice_queries_;
  }

  const std::vector<InstanceQueryData>& instance_queries() const {
    return instance_queries_;
  }

  const std::vector<FixedQueryData>& fixed_queries() const {
    return fixed_queries_;
  }

  const PermutationArgument& permutation() const { return permutation_; }

  const std::vector<lookup::Argument<F>>& lookups() const { return lookups_; }

  const absl::btree_map<std::string, LookupTracker<F>>& lookups_map() const {
    return lookups_map_;
  }

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
  void set_minimum_degree(size_t degree) {
    if (minimum_degree_) {
      minimum_degree_ = std::max(minimum_degree_.value(), degree);
    } else {
      minimum_degree_ = degree;
    }
  }

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
    QueryAnyIndex(column, Rotation::Cur());
    permutation_.AddColumn(column);
  }

  // Add a lookup argument for some input expressions and table columns.
  //
  // |callback| returns a map between the input expressions and the table
  // columns they need to match.
  void Lookup(std::string_view name, LookupCallback callback) {
    VirtualCells cells(this);
    lookup::Pairs<std::unique_ptr<Expression<F>>, LookupTableColumn> pairs =
        std::move(callback).Run(cells);

    switch (lookup_type_) {
      case lookup::Type::kHalo2: {
        lookup::Pairs<std::unique_ptr<Expression<F>>> lookup_pairs = base::Map(
            pairs, [&cells](lookup::Pair<std::unique_ptr<Expression<F>>,
                                         LookupTableColumn>& pair) {
              std::unique_ptr<Expression<F>> table =
                  cells.QueryLookupTable(pair, Rotation::Cur());
              return lookup::Pair<std::unique_ptr<Expression<F>>>(
                  std::move(pair).TakeInput(), std::move(table));
            });
        lookups_.emplace_back(name, std::move(lookup_pairs));
        break;
      }
      case lookup::Type::kLogDerivativeHalo2: {
        std::vector<std::unique_ptr<Expression<F>>> input_expressions;
        std::vector<std::unique_ptr<Expression<F>>> table_expressions;

        input_expressions.reserve(pairs.size());
        table_expressions.reserve(pairs.size());

        for (lookup::Pair<std::unique_ptr<Expression<F>>, LookupTableColumn>&
                 pair : pairs) {
          std::unique_ptr<Expression<F>> table =
              cells.QueryLookupTable(pair, Rotation::Cur());
          input_expressions.push_back(std::move(pair).TakeInput());
          table_expressions.push_back(std::move(table));
        }
        UpdateLookupsMap(name, std::move(input_expressions),
                         std::move(table_expressions));
        break;
      }
    }
  }

  // Add a lookup argument for some input expressions and table expressions.
  //
  // |callback| returns a map between the input expressions and the table
  // expressions they need to match.
  void LookupAny(std::string_view name, LookupAnyCallback callback) {
    VirtualCells cells(this);
    lookup::Pairs<std::unique_ptr<Expression<F>>> pairs =
        std::move(callback).Run(cells);

    switch (lookup_type_) {
      case lookup::Type::kHalo2: {
        for (const lookup::Pair<std::unique_ptr<Expression<F>>>& pair : pairs) {
          CHECK(!ContainsSimpleSelector(pair.input().get()))
              << "expression containing simple selector "
                 "supplied to lookup argument";
        }
        lookups_.emplace_back(name, std::move(pairs));
        break;
      }
      case lookup::Type::kLogDerivativeHalo2: {
        std::vector<std::unique_ptr<Expression<F>>> input_expressions;
        std::vector<std::unique_ptr<Expression<F>>> table_expressions;

        input_expressions.reserve(pairs.size());
        table_expressions.reserve(pairs.size());

        for (lookup::Pair<std::unique_ptr<Expression<F>>>& pair : pairs) {
          CHECK(!ContainsSimpleSelector(pair.input().get()))
              << "expression containing simple selector "
                 "supplied to lookup argument";

          input_expressions.push_back(std::move(pair).TakeInput());
          table_expressions.push_back(std::move(pair).TakeTable());
        }

        UpdateLookupsMap(name, std::move(input_expressions),
                         std::move(table_expressions));
        break;
      }
    }
  }

  // Chunk lookup arguments into pieces below a given degree bound. Compute the
  // |minimum_degree| from gates and lookups, and then construct the
  // |lookup_arguments| of chunks smaller than the |minimum_degree| from the
  // |lookups_map|.
  void ChunkLookups() {
    CHECK_EQ(lookup_type_, lookup::Type::kLogDerivativeHalo2);

    if (lookups_map_.empty()) {
      return;
    }

    size_t max_gate_degree = ComputeGateRequiredDegree();

    size_t max_single_lookup_degree = 0;
    for (const auto& [_, lookup_tracker] : lookups_map_) {
      size_t table_degree = ComputeColumnDegree(lookup_tracker.table);

      // Compute base degree only from table without inputs.
      size_t base_lookup_degree = ComputeBaseDegree(table_degree);

      size_t max_inputs_degree = 0;
      for (const std::vector<std::unique_ptr<Expression<F>>>& input :
           lookup_tracker.inputs) {
        max_inputs_degree =
            std::max(max_inputs_degree, ComputeColumnDegree(input));
      }

      size_t current_degree =
          ComputeDegreeWithInput(base_lookup_degree, max_inputs_degree);
      max_single_lookup_degree =
          std::max(max_single_lookup_degree, current_degree);
    }

    size_t required_degree =
        std::max(max_gate_degree, max_single_lookup_degree);

    // The smallest power of 2 greater than or equal to |required_degree|.
    size_t next_power_of_two = absl::bit_ceil(required_degree);

    set_minimum_degree(next_power_of_two + 1);

    size_t minimum_degree = minimum_degree_.value();

    for (const auto& [_, lookup_tracker] : lookups_map_) {
      std::vector<std::unique_ptr<Expression<F>>> cloned_input =
          Expression<F>::CloneExpressions(lookup_tracker.inputs[0]);
      std::vector<std::unique_ptr<Expression<F>>> cloned_table =
          Expression<F>::CloneExpressions(lookup_tracker.table);
      lookups_.emplace_back(lookup_tracker.name, std::move(cloned_input),
                            std::move(cloned_table));

      for (auto input = lookup_tracker.inputs.begin() + 1;
           input != lookup_tracker.inputs.end(); ++input) {
        // Compute the degree of the current set of input expressions.
        size_t cur_input_degree = ComputeColumnDegree(*input);

        bool added = false;
        for (lookup::Argument<F>& lookup_argument : lookups_) {
          // Try to fit input into one of the |lookup_arguments|.
          size_t cur_argument_degree = lookup_argument.RequiredDegree();
          size_t new_potential_degree = cur_argument_degree + cur_input_degree;
          if (new_potential_degree <= minimum_degree) {
            std::vector<std::unique_ptr<Expression<F>>> cloned_input =
                Expression<F>::CloneExpressions(*input);
            lookup_argument.inputs_expressions().push_back(
                std::move(cloned_input));
            added = true;
            break;
          }
        }

        if (!added) {
          std::vector<std::unique_ptr<Expression<F>>> cloned_input =
              Expression<F>::CloneExpressions(*input);
          std::vector<std::unique_ptr<Expression<F>>> cloned_table =
              Expression<F>::CloneExpressions(lookup_tracker.table);
          lookups_.emplace_back(lookup_tracker.name, std::move(cloned_input),
                                std::move(cloned_table));
        }
      }
    }
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
    ++num_advice_queries_[column.index()];
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
  // crash if |constrain| returns an empty iterator.
  void CreateGate(std::string_view name, ConstrainCallback constrain) {
    VirtualCells cells(this);
    std::vector<Constraint<F>> constraints = std::move(constrain).Run(cells);
    std::vector<Selector> queried_selectors =
        std::move(cells).TakeQueriedSelectors();
    std::vector<VirtualCell> queried_cells =
        std::move(cells).TakeQueriedCells();

    std::vector<std::string> constraint_names;
    std::vector<std::unique_ptr<Expression<F>>> polys;
    for (Constraint<F>& constraint : constraints) {
      constraint_names.push_back(std::move(constraint).TakeName());
      polys.push_back(std::move(constraint).TakeExpression());
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
    std::vector<size_t> degrees(selectors.size(), size_t{0});
    for (const Gate<F>& gate : gates_) {
      for (const std::unique_ptr<Expression<F>>& expression : gate.polys()) {
        std::optional<Selector> selector =
            ExtractSimpleSelector(expression.get());
        if (selector.has_value()) {
          degrees[selector->index()] =
              std::max(degrees[selector->index()], expression->Degree());
        }
      }
    }

    // We will not increase the degree of the constraint system, so we limit
    // ourselves to the largest existing degree constraint.
    std::vector<FixedColumnKey> new_columns;
    SelectorCompressor<F> selector_compressor;
    selector_compressor.Process(
        selectors, degrees, ComputeDegree(), [this, &new_columns]() {
          FixedColumnKey column = CreateFixedColumn();
          new_columns.push_back(column);
          return ExpressionFactory<F>::Fixed(
              FixedQuery(QueryFixedIndex(column, Rotation::Cur()),
                         Rotation::Cur(), FixedColumnKey(column.index())));
        });

    std::vector<FixedColumnKey> selector_map;
    selector_map.resize(selector_compressor.selector_assignments().size());
    std::vector<base::Ref<const Expression<F>>> selector_replacements(
        selector_compressor.selector_assignments().size(),
        base::Ref<const Expression<F>>());
    for (const SelectorAssignment<F>& assignment :
         selector_compressor.selector_assignments()) {
      selector_replacements[assignment.selector_index()] =
          base::Ref<const Expression<F>>(assignment.expression());
      selector_map[assignment.selector_index()] =
          new_columns[assignment.combination_index()];
    }

    selector_map_ = std::move(selector_map);

    for (Gate<F>& gate : gates_) {
      for (std::unique_ptr<Expression<F>>& expression : gate.polys()) {
        expression =
            ReplaceSelectors(expression.get(), selector_replacements, false);
      }
    }
    for (lookup::Argument<F>& lookup : lookups_) {
      for (std::vector<std::unique_ptr<Expression<F>>>& input_expressions :
           lookup.inputs_expressions()) {
        for (std::unique_ptr<Expression<F>>& expression : input_expressions) {
          expression =
              ReplaceSelectors(expression.get(), selector_replacements, true);
        }
      }
      for (std::unique_ptr<Expression<F>>& expression :
           lookup.table_expressions()) {
        expression =
            ReplaceSelectors(expression.get(), selector_replacements, true);
      }
    }

    return selector_compressor.combination_assignments();
  }

  // Allocate a new simple selector. Simple selectors cannot be added to
  // expressions nor multiplied by other expressions containing simple
  // selectors. Also, simple selectors may not appear in lookup argument
  // inputs.
  Selector CreateSimpleSelector() {
    ++num_simple_selectors_;
    return Selector::Simple(num_selectors_++);
  }

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
          << "Phase " << previous_phase.ToString() << " is not used";
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
        << "Phase " << phase.ToString() << " is not used";
    Challenge challenge(num_challenges_++, phase);
    challenge_phases_.push_back(phase);
    return challenge;
  }

  Phase ComputeMaxPhase() const {
    auto max_phase_it = std::max_element(advice_column_phases_.begin(),
                                         advice_column_phases_.end());
    if (max_phase_it == advice_column_phases_.end()) return kFirstPhase;
    return *max_phase_it;
  }

  std::vector<Phase> GetPhases() const {
    Phase max_phase = ComputeMaxPhase();
    return base::CreateVector(
        static_cast<size_t>(max_phase.value() + 1),
        [](size_t i) { return Phase(static_cast<uint8_t>(i)); });
  }

  // Compute the degree of the constraint system (the maximum degree of all
  // constraints).
  size_t ComputeDegree() const {
    if (!cached_degree_.has_value()) {
      // The permutation argument will serve alongside the gates, so must be
      // accounted for.
      size_t degree = permutation_.RequiredDegree();

      // The lookup argument also serves alongside the gates and must be
      // accounted for.
      degree = std::max(degree, ComputeLookupRequiredDegree());

      // Account for each gate to ensure our quotient polynomial is the
      // correct degree and that our extended domain is the right size.
      degree = std::max(degree, ComputeGateRequiredDegree());

      cached_degree_ = std::max(degree, minimum_degree_.value_or(1));
    }
    return *cached_degree_;
  }

  // Compute the extended K. Since the degree of h(X) exceeds 2ᵏ - 1, h(X) will
  // be split to h₀(X), h₁(X), ..., h_{d - 1}(X). The extended K denotes the K
  // of the extended domain for these.
  //
  // h(X) = (gate₀(X) + y * gate₁(X) + ... + yⁱ * gateᵢ(X) + ...) / t(X)
  //
  // h₀(X) + Xⁿ * h₁(X) + ... + Xⁿ⁽ᵈ⁻¹⁾ * h_{d - 1}(X).
  // H = [Commit(h₀(X)), Commit(h₁(X)), ..., Commit(h_{d - 1}(X))]
  //
  // See
  // https://zcash.github.io/halo2/design/proving-system/vanishing.html#committing-to-hx.
  uint32_t ComputeExtendedK(uint32_t k) const {
    size_t quotient_poly_degree = ComputeDegree() - 1;
    return std::max(
        base::bits::SafeLog2Ceiling((size_t{1} << k) * quotient_poly_degree),
        k);
  }

  // Compute the number of blinding factors necessary to perfectly blind
  // each of the prover's witness polynomials.
  RowIndex ComputeBlindingFactors() const {
    if (!cached_blinding_factors_.has_value()) {
      // All of the prover's advice columns are evaluated at no more than
      auto max_num_advice_query_it = std::max_element(
          num_advice_queries_.begin(), num_advice_queries_.end());

      RowIndex factors = max_num_advice_query_it == num_advice_queries_.end()
                             ? 1
                             : *max_num_advice_query_it;
      // distinct points during gate checks.

      // - The permutation argument witness polynomials are evaluated at most 3
      //   times.
      // - Each lookup argument has independent witness polynomials, and they
      //   are evaluated at most 2 times.
      factors = std::max(RowIndex{3}, factors);
      CHECK_LE(factors, UINT32_MAX - 2);

      // Each polynomial is evaluated at most an additional time during
      // multiopen (at x₃ to produce q_evals):
      ++factors;

      // h(x) is derived by the other evaluations so it does not reveal
      // anything; in fact it does not even appear in the proof.

      // h(x₃) is also not revealed; the verifier only learns a single
      // evaluation of a polynomial in x₁ which has h(x₃) and another random
      // polynomial evaluated at x₃ as coefficients -- this random polynomial
      // is "random_poly" in the vanishing argument.

      // Add an additional blinding factor as a slight defense against
      // off-by-one errors.
      cached_blinding_factors_ = ++factors;
    }
    return *cached_blinding_factors_;
  }

  RowOffset ComputeLastRow() const { return -(ComputeBlindingFactors() + 1); }

  // Returns the minimum necessary rows that need to exist in order to
  // account for e.g. blinding factors.
  RowIndex ComputeMinimumRows() const {
    return ComputeBlindingFactors()  // m blinding factors
           + 1                       // for l_{-(m + 1)} (l_last)
           +
           1  // for l_first (just for extra breathing room for the permutation
              // argument, to essentially force a separation in the
           // permutation polynomial between the roles of l_last, l_first
           // and the interstitial values.)
           + 1;  // for at least one row
  }

  // Return the length of permutation chunk.
  size_t ComputePermutationChunkLen() const {
    return ComputePermutationChunkLength(ComputeDegree());
  }

  // Return the number of permutation products.
  size_t ComputePermutationProductNums() const {
    size_t chunk_len = ComputePermutationChunkLen();
    return (permutation_.columns().size() + chunk_len - 1) / chunk_len;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "num_fixed_columns: " << num_fixed_columns_
       << ", num_advice_columns: " << num_advice_columns_
       << ", num_instance_columns: " << num_instance_columns_
       << ", num_simple_selectors: " << num_simple_selectors_
       << ", num_selectors: " << num_selectors_
       << ", num_challenges: " << num_challenges_
       << ", degree: " << ComputeDegree()
       << ", minimum_degree: " << base::OptionalToString(minimum_degree_)
       << ", blinding_factors: " << ComputeBlindingFactors()
       << ", max_phase: " << uint32_t{ComputeMaxPhase().value()}
       << ", permutations: " << permutation_.columns().size()
       << ", lookups: " << lookups_.size();
    return ss.str();
  }

 private:
  template <typename LS>
  friend class c::zk::plonk::ProvingKeyImpl;

  FRIEND_TEST(ConstraintSystemTest, Lookup);
  FRIEND_TEST(ConstraintSystemTest, LookupAny);

  void UpdateLookupsMap(
      std::string_view name,
      std::vector<std::unique_ptr<Expression<F>>>&& input_expressions,
      std::vector<std::unique_ptr<Expression<F>>>&& table_expressions) {
    std::stringstream table_expressions_ss;
    for (const std::unique_ptr<Expression<F>>& expr : table_expressions) {
      table_expressions_ss << Identifier(expr.get());
    }

    std::string table_expressions_identifier = table_expressions_ss.str();

    auto it = lookups_map_.find(table_expressions_identifier);
    if (it != lookups_map_.end()) {
      it->second.inputs.push_back(std::move(input_expressions));
    } else {
      LookupTracker<F> lookup_tracker(name, std::move(table_expressions),
                                      std::move(input_expressions));
      lookups_map_[table_expressions_identifier] = std::move(lookup_tracker);
    }
  }

  template <typename QueryData, typename Column>
  static bool QueryIndex(const std::vector<QueryData>& queries,
                         const Column& column, Rotation at, size_t* index_out) {
    std::optional<size_t> index =
        base::FindIndexIf(queries, [&column, at](const QueryData& query) {
          return query.column() == column && query.rotation() == at;
        });
    if (!index.has_value()) return false;
    *index_out = index.value();
    return true;
  }

  static size_t ComputeColumnDegree(
      const std::vector<std::unique_ptr<Expression<F>>>& column) {
    return (*std::max_element(column.begin(), column.end(),
                              [](const std::unique_ptr<Expression<F>>& a,
                                 const std::unique_ptr<Expression<F>>& b) {
                                return a->Degree() < b->Degree();
                              }))
        ->Degree();
  }

  size_t ComputeLookupRequiredDegree() const {
    std::vector<size_t> required_degrees =
        base::Map(lookups_, [](const lookup::Argument<F>& argument) {
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
          return base::Map(gate.polys(),
                           [](const std::unique_ptr<Expression<F>>& poly) {
                             return poly->Degree();
                           });
        });
    auto max_required_degree =
        std::max_element(required_degrees.begin(), required_degrees.end());
    if (max_required_degree == required_degrees.end()) return 0;
    return *max_required_degree;
  }

  // Degree of lookup without inputs
  constexpr static size_t ComputeBaseDegree(size_t table_degree) {
    // φᵢ(X) = fᵢ(X) + α
    // τ(X) = t(X) + α
    //
    // LHS = τ(X) * Π(φᵢ(X)) * (ϕ(gX) - ϕ(X))
    // ↪ DEG(LHS) = |table_degree| + inputs_degree(= 0) + 1
    //            = |table_degree| + 1
    //
    // RHS = τ(X) * Π(φᵢ(X)) * (∑ 1/(φᵢ(X)) - m(X) / τ(X))
    // ↪ DEG(RHS) = |table_degree|
    //
    // (1 - (l_last(X) + l_blind(X))) * (LHS - RHS)
    // ↪ degree = DEG(LHS) + 1 = |table_degree| + 2
    return std::max(size_t{3}, table_degree + 2);
  }

  constexpr static size_t ComputeDegreeWithInput(
      size_t base_degree, size_t input_expression_degree) {
    return base_degree + input_expression_degree;
  }

  // TODO(Insun35): Change default |lookup_type_| to |kLogDerivativeHalo2|
  // after implementing C API for LogDerivativeHalo2 scheme.
  lookup::Type lookup_type_ = lookup::Type::kHalo2;

  size_t num_fixed_columns_ = 0;
  size_t num_advice_columns_ = 0;
  size_t num_instance_columns_ = 0;
  size_t num_simple_selectors_ = 0;
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
  std::vector<RowIndex> num_advice_queries_;
  std::vector<InstanceQueryData> instance_queries_;
  std::vector<FixedQueryData> fixed_queries_;

  // Permutation argument for performing equality constraints.
  PermutationArgument permutation_;

  // Vector of lookup arguments, where each corresponds
  // to a sequence of input expressions and a sequence
  // of table expressions involved in the lookup.
  std::vector<lookup::Argument<F>> lookups_;

  // NOTE(Insun35): |lookups_map_| is only used for LogDerivativeHalo2.
  // btree_map with |table_expressions_identifier| as a key and |LookupTracker|
  // as values.
  absl::btree_map<std::string, LookupTracker<F>> lookups_map_;

  // List of indexes of Fixed columns which are associated to a
  // circuit-general Column tied to their annotation.
  absl::flat_hash_map<ColumnKeyBase, std::string> general_column_annotations_;

  // Vector of fixed columns, which can be used to store constant values
  // that are copied into advice columns.
  std::vector<FixedColumnKey> constants_;

  std::optional<size_t> minimum_degree_;

  mutable std::optional<size_t> cached_degree_;
  mutable std::optional<RowIndex> cached_blinding_factors_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
