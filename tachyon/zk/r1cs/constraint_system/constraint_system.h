// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_

#include <stddef.h>

#include <optional>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/r1cs/constraint_system/constraint_matrices.h"
#include "tachyon/zk/r1cs/constraint_system/linear_combination.h"
#include "tachyon/zk/r1cs/constraint_system/optimization_goal.h"
#include "tachyon/zk/r1cs/constraint_system/synthesis_mode.h"

namespace tachyon::zk::r1cs {

// A Rank-One |ConstraintSystem|. Enforces constraints of the form
// ⟨aᵢ, z⟩ ⋅ ⟨bᵢ, z⟩ = ⟨cᵢ, z⟩, where aᵢ, bᵢ, and cᵢ are linear
// combinations over variables, and z is the concrete assignment to these
// variables.
template <typename F>
class ConstraintSystem {
 public:
  using CreateVariableCallback = base::OnceCallback<F()>;

  struct TransformLCMapResult {
    size_t num_new_witness_variables_needed;
    std::optional<std::vector<F>> new_witness_assignments;

    explicit TransformLCMapResult(size_t num_new_witness_variables_needed)
        : num_new_witness_variables_needed(num_new_witness_variables_needed) {}
    TransformLCMapResult(size_t num_new_witness_variables_needed,
                         std::vector<F>&& new_witness_assignments)
        : num_new_witness_variables_needed(num_new_witness_variables_needed),
          new_witness_assignments(std::move(new_witness_assignments)) {}
  };

  using TransformLCMapCallback = base::RepeatingCallback<TransformLCMapResult(
      size_t, LinearCombination<F>&)>;

  ConstraintSystem() = default;

  SynthesisMode mode() const { return mode_; }
  void set_mode(SynthesisMode mode) { mode_ = mode; }
  size_t num_instance_variables() const { return num_instance_variables_; }
  size_t num_witness_variables() const { return num_witness_variables_; }
  size_t num_constraints() const { return num_constraints_; }
  size_t num_linear_combinations() const { return num_linear_combinations_; }
  OptimizationGoal optimization_goal() const { return optimization_goal_; }
  // Specify whether this constraint system should aim to optimize weight,
  // number of constraints, or neither.
  void set_optimization_goal(OptimizationGoal optimization_goal) {
    // |set_optimization_goal()| should only be executed before any constraint
    // or value is created.
    CHECK_EQ(num_instance_variables_, size_t{1});
    CHECK_EQ(num_witness_variables_, size_t{0});
    CHECK_EQ(num_constraints_, size_t{0});
    CHECK_EQ(num_linear_combinations_, size_t{0});
    optimization_goal_ = optimization_goal;
  }
  const std::vector<F>& instance_assignments() const {
    return instance_assignments_;
  }
  const std::vector<F>& witness_assignments() const {
    return witness_assignments_;
  }

  std::vector<Cell<F>> MakeRow(const LinearCombination<F>& lc) const {
    std::vector<Cell<F>> row;
    for (const Term<F>& term : lc.terms()) {
      if (!term.coefficient.IsZero()) {
        row.push_back({term.coefficient,
                       *term.variable.GetIndex(num_instance_variables_)});
      }
    }
    return row;
  }

  // Return a variable representing the constant "zero" inside the constraint
  // system.
  Variable CreateZero() const { return Variable::Zero(); }

  // Return a variable representing the constant "one" inside the constraint
  // system.
  Variable CreateOne() const { return Variable::One(); }

  // Obtain a variable representing a new public instance input.
  Variable CreateInstanceVariable(CreateVariableCallback callback) {
    if (!mode_.IsSetup()) {
      instance_assignments_.push_back(std::move(callback).Run());
    }
    return Variable::Instance(num_instance_variables_++);
  }

  // Obtain a variable representing a new private witness input.
  Variable CreateWitnessVariable(CreateVariableCallback callback) {
    if (!mode_.IsSetup()) {
      witness_assignments_.push_back(std::move(callback).Run());
    }
    return Variable::Witness(num_witness_variables_++);
  }

  // Obtain a variable representing a linear combination.
  Variable CreateLinearCombination(
      const LinearCombination<F>& linear_combination) {
    size_t index = num_linear_combinations_++;
    lc_map_[index] = linear_combination;
    return Variable::SymbolicLinearCombination(index);
  }

  // Obtain a variable representing a linear combination.
  Variable CreateLinearCombination(LinearCombination<F>&& linear_combination) {
    size_t index = num_linear_combinations_++;
    lc_map_[index] = std::move(linear_combination);
    return Variable::SymbolicLinearCombination(index);
  }

  // Enforce a constraint that a * b = c. It terminates when
  // |mode_.ShouldConstructMatrices()| is false.
  void EnforceConstraint(const LinearCombination<F>& a,
                         const LinearCombination<F>& b,
                         const LinearCombination<F>& c) {
    CHECK(mode_.ShouldConstructMatrices());
    size_t a_index = CreateLinearCombination(a).index();
    size_t b_index = CreateLinearCombination(b).index();
    size_t c_index = CreateLinearCombination(c).index();
    a_constraints_.push_back(a_index);
    b_constraints_.push_back(b_index);
    c_constraints_.push_back(c_index);
    ++num_constraints_;
    // TODO(chokobole): add constraint trace
  }

  // Enforce a constraint that a * b = c. It terminates when
  // |mode_.ShouldConstructMatrices()| is false.
  void EnforceConstraint(LinearCombination<F>&& a, LinearCombination<F>&& b,
                         LinearCombination<F>&& c) {
    CHECK(mode_.ShouldConstructMatrices());
    size_t a_index = CreateLinearCombination(std::move(a)).index();
    size_t b_index = CreateLinearCombination(std::move(b)).index();
    size_t c_index = CreateLinearCombination(std::move(c)).index();
    a_constraints_.push_back(a_index);
    b_constraints_.push_back(b_index);
    c_constraints_.push_back(c_index);
    ++num_constraints_;
    // TODO(chokobole): add constraint trace
  }

  // Count the number of times each LC is used within other LCs in the
  // constraint system
  std::vector<size_t> ComputeLCNumTimesUsed(bool count_sinks) const {
    std::vector<size_t> num_times_used;
    num_times_used.resize(lc_map_.size());

    for (const std::pair<const size_t, LinearCombination<F>>& entry : lc_map_) {
      num_times_used[entry.first] += size_t{count_sinks};

      // Increment the counter for each lc that this lc has a direct
      // dependency on.
      for (const Term<F>& term : entry.second.terms()) {
        if (term.variable.IsSymbolicLinearCombination()) {
          ++num_times_used[term.variable.index()];
        }
      }
    }
    return num_times_used;
  }

  // Transform the map of linear combinations.
  // Specifically, allow the creation of additional witness assignments.
  //
  // This method is used as a subroutine of |InlineAllLCs| and |OutlineLCs|.
  void TransformLCMap(TransformLCMapCallback callback) {
    absl::btree_map<size_t, LinearCombination<F>> transformed_lc_map;
    std::vector<size_t> num_times_used = ComputeLCNumTimesUsed(false);

    // This loop goes through all the LCs in the map, starting from
    // the early ones. The transformer function is applied to the
    // inlined LC, where new witness variables can be created.
    for (std::pair<const size_t, LinearCombination<F>>& entry : lc_map_) {
      LinearCombination<F> transformed_lc;

      // Inline the LC, unwrapping symbolic LCs that may constitute it,
      // and updating them according to transformations in prior iterations.
      for (Term<F>& term : entry.second.terms()) {
        if (term.variable.IsSymbolicLinearCombination()) {
          size_t lc_index = term.variable.index();

          // If |term.variable| is a |SymbolicLinearCombination|, fetch the
          // corresponding inlined LC, and substitute it in.
          //
          // We have the guarantee that |lc_index| must exist in
          // |new_lc_map| since a LC can only depend on other
          // LCs with lower indices, which we have transformed.
          //
          auto it = transformed_lc_map.find(lc_index);
          LinearCombination<F>& lc = it->second;
          transformed_lc.AppendTerms((lc * term.coefficient).TakeTerms());

          // Delete linear combinations that are no longer used.
          //
          // Deletion is safe for both outlining and inlining:
          // * Inlining: the LC is substituted directly into all use sites, and
          //   so once it is fully inlined, it is redundant.
          //
          // * Outlining: the LC is associated with a new variable |w|, and a
          //   new constraint of the form |lc_data * 1 = w|, where |lc_data| is
          //   the actual data in the linear combination. Furthermore, we
          //   replace its entry in |new_lc_map| with |(1, w)|. Once |w| is
          //   fully inlined, then we can delete the entry from |new_lc_map|
          //
          if (--num_times_used[lc_index] == 0) {
            transformed_lc_map.erase(it);
          }
        } else {
          // Otherwise, it's a concrete variable and so we
          // substitute it in directly.
          transformed_lc.AppendTerm(std::move(term));
        }
      }
      transformed_lc.Deduplicate();

      // Call the transformer function.
      TransformLCMapResult result =
          callback.Run(num_times_used[entry.first], transformed_lc);

      // Insert the transformed LC.
      transformed_lc_map[entry.first] = std::move(transformed_lc);

      // Update the witness counter.
      num_witness_variables_ += result.num_new_witness_variables_needed;

      // Supply additional witness assignments if not in the
      // setup mode and if new witness variables are created.
      if (!mode_.IsSetup() && result.num_new_witness_variables_needed > 0) {
        CHECK(result.new_witness_assignments.has_value());
        CHECK_EQ(result.new_witness_assignments->size(),
                 result.num_new_witness_variables_needed);
        base::Extend(witness_assignments_,
                     std::move(*result.new_witness_assignments));
      }
    }
    lc_map_ = transformed_lc_map;
  }

  // Naively inlines symbolic linear combinations into the linear
  // combinations that use them.
  //
  // Useful for standard pairing-based SNARKs where addition gates are cheap.
  // For example, in the SNARKs such as
  // [Groth16](https://eprint.iacr.org/2016/260) and
  // [Groth-Maller17](https://eprint.iacr.org/2017/540), addition gates
  // do not contribute to the size of the multi-scalar multiplication, which
  // is the dominating cost.
  void InlineAllLCs() {
    // Only inline when a matrix representing R1CS is needed.
    if (!mode_.ShouldConstructMatrices()) return;

    // A dummy closure is used, which means that
    // - it does not modify the |inlined_lc|.
    // - it does not add new witness variables.
    TransformLCMap([](size_t, LinearCombination<F>& inlined_lc) {
      return TransformLCMapResult(0);
    });
  }

  // If a |SymbolicLinearCombination| is used in more than one location and has
  // sufficient length, this method makes a new variable for that
  // |SymbolicLinearCombination|, adds a constraint ensuring the equality of the
  // variable and the linear combination, and then uses that variable in every
  // location the |SymbolicLinearCombination| is used.
  //
  // Useful for SNARKs like [Marlin](https://eprint.iacr.org/2019/1047) or
  // [Fractal](https://eprint.iacr.org/2019/1076), where addition gates
  // are not cheap.
  void OutlineLCs() {
    // Only inline when a matrix representing R1CS is needed.
    if (!mode_.ShouldConstructMatrices()) return;

    // Store information about new witness variables created
    // for outlining. New constraints will be added after the
    // transformation of the LC map.
    std::vector<LinearCombination<F>> new_witness_linear_combinations;
    std::vector<size_t> new_witness_indices;

    // It goes through all the LCs in the map, starting from
    // the early ones, and decides whether or not to dedicate a witness
    // variable for this LC.
    //
    // If true, the LC is replaced with 1 * this witness variable.
    // Otherwise, the LC is inlined.
    //
    // Each iteration first updates the LC according to outlinings in prior
    // iterations, and then sees if it should be outlined, and if so adds
    // the outlining to the map.
    TransformLCMap([this, &new_witness_linear_combinations,
                    &new_witness_indices](size_t num_times_used,
                                          LinearCombination<F>& inlined_lc) {
      bool should_dedicate_a_witness_variable = false;
      std::optional<size_t> new_witness_index;
      std::vector<F> new_witness_assignments;

      // Check if it is worthwhile to dedicate a witness variable.
      size_t this_used_times = num_times_used + 1;
      size_t this_len = inlined_lc.terms().size();

      // Cost with no outlining = |this_len * this_used_times|
      // Cost with outlining is one constraint for |(this_len) * 1 = {new
      // variable}| and using that single new variable in each of the prior
      // usages. This has total cost |this_used_times + this_len + 2|
      if (this_len * this_used_times > this_used_times + this_len + 2) {
        should_dedicate_a_witness_variable = true;
      }

      // If it is worthwhile to dedicate a witness variable,
      if (should_dedicate_a_witness_variable) {
        // Add a new witness (the value of the linear combination).
        // This part follows the same logic of |new_witness_variable_|.
        size_t witness_index = num_witness_variables_;
        new_witness_index = witness_index;

        // Compute the witness assignment.
        if (!mode_.IsSetup()) {
          new_witness_assignments.push_back(
              DoEvalLinearCombination(inlined_lc));
        }

        // Add a new constraint for this new witness.
        new_witness_linear_combinations.push_back(std::move(inlined_lc));
        new_witness_indices.push_back(witness_index);

        // Replace the linear combination with (1 * this new witness).
        inlined_lc =
            LinearCombination<F>({Term<F>(Variable::Witness(witness_index))});
      }
      // Otherwise, the LC remains unchanged.

      // Return information about new witness variables.
      if (new_witness_index.has_value()) {
        return TransformLCMapResult(1, std::move(new_witness_assignments));
      } else {
        return TransformLCMapResult(0);
      }
    });

    for (size_t i = 0; i < new_witness_linear_combinations.size(); ++i) {
      EnforceConstraint(new_witness_linear_combinations[i],
                        LinearCombination<F>({Term<F>(CreateOne())}),
                        LinearCombination<F>({Term<F>(
                            Variable::Witness(new_witness_indices[i]))}));
    }
  }

  // Finalize the constraint system (either by outlining or inlining,
  // if an optimization goal is set).
  void Finalize() {
    switch (optimization_goal_) {
      case OptimizationGoal::kConstraints:
        InlineAllLCs();
        return;
      case OptimizationGoal::kWeight:
        OutlineLCs();
        return;
    }
    NOTREACHED();
  }

  // This step must be called after constraint generation has completed, and
  // after all symbolic LCs have been inlined into the places that they
  // are used.
  std::optional<ConstraintMatrices<F>> ToMatrices() const {
    if (!mode_.ShouldConstructMatrices()) return std::nullopt;
    Matrix<F> a = MakeMatrix(a_constraints_);
    Matrix<F> b = MakeMatrix(b_constraints_);
    Matrix<F> c = MakeMatrix(c_constraints_);
    ConstraintMatrices<F> matrices{num_instance_variables_,
                                   num_witness_variables_,
                                   num_constraints_,
                                   a.CountNonZero(),
                                   b.CountNonZero(),
                                   c.CountNonZero(),
                                   std::move(a),
                                   std::move(b),
                                   std::move(c)};
    return matrices;
  }

  // Evaluate the linear combination corresponding to the |index|.
  F EvalLinearCombination(size_t index) const {
    auto it = lc_map_.find(index);
    CHECK(it != lc_map_.end());
    const LinearCombination<F>& lc = it->second;
    return DoEvalLinearCombination(lc);
  }

  // Obtain the assignment corresponding to the |v|.
  F GetAssignedValue(const Variable& v) const {
    switch (v.type()) {
      case Variable::Type::kZero:
        return F::Zero();
      case Variable::Type::kOne:
        return F::One();
      case Variable::Type::kInstance:
        return instance_assignments_[v.index()];
      case Variable::Type::kWitness:
        return witness_assignments_[v.index()];
      case Variable::Type::kSymbolicLinearCombination: {
        auto it = lc_assignment_cache_.find(v.index());
        if (it == lc_assignment_cache_.end()) {
          it = lc_assignment_cache_
                   .insert({v.index(), EvalLinearCombination(v.index())})
                   .first;
        }
        return it->second;
      }
    }
    NOTREACHED();
    return F::Zero();
  }

  // Return true if |*this| is satisfied.
  // Return false if |*this| is not satisfied or |mode_.IsSetup()| is true.
  bool IsSatisfied() const {
    if (mode_.IsSetup()) {
      LOG(ERROR) << "Assignments are missing";
      return false;
    }
    for (size_t i = 0; i < num_constraints_; ++i) {
      F a = EvalLinearCombination(a_constraints_[i]);
      F b = EvalLinearCombination(b_constraints_[i]);
      F c = EvalLinearCombination(c_constraints_[i]);
      if (a * b != c) return false;
    }
    return true;
  }

 private:
  FRIEND_TEST(ConstraintSystemTest, GetAssignedValue);

  Matrix<F> MakeMatrix(const std::vector<size_t>& constraints) const {
    return Matrix<F>(base::Map(constraints, [this](size_t index) {
      auto it = lc_map_.find(index);
      return MakeRow(it->second);
    }));
  }

  F DoEvalLinearCombination(const LinearCombination<F>& lc) const {
    return std::accumulate(lc.terms().begin(), lc.terms().end(), F::Zero(),
                           [this](F& acc, const Term<F>& term) {
                             return acc += term.coefficient *
                                           GetAssignedValue(term.variable);
                           });
  }

  // The mode in which the constraint system is operating. |this| can either
  // be in setup mode or in proving mode. If we are in proving mode, then
  // we have the additional option of whether or not to construct the A, B,
  // and C matrices of the constraint system (see below).
  SynthesisMode mode_ = SynthesisMode::Prove(true);
  // The number of variables that are "public inputs" to the
  // constraint system.
  size_t num_instance_variables_ = 1;
  // The number of variables that are "private inputs" to the constraint system.
  size_t num_witness_variables_ = 0;
  // The number of constraints in the constraint system.
  size_t num_constraints_ = 0;
  // The number of linear combinations
  size_t num_linear_combinations_ = 0;

  // The parameter we aim to minimize in this constraint system (either the
  // number of constraints or their total weight).
  OptimizationGoal optimization_goal_ = OptimizationGoal::kConstraints;

  // Assignments to the public input variables. This is empty in setup mode.
  std::vector<F> instance_assignments_ = {F::One()};
  // Assignments to the private input variables. This is empty in setup mode.
  std::vector<F> witness_assignments_;

  absl::btree_map<size_t, LinearCombination<F>> lc_map_;

  std::vector<size_t> a_constraints_;
  std::vector<size_t> b_constraints_;
  std::vector<size_t> c_constraints_;

  mutable absl::btree_map<size_t, F> lc_assignment_cache_;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
