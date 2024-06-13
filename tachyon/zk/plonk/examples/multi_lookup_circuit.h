#ifndef TACHYON_ZK_PLONK_EXAMPLES_MULTI_LOOKUP_CIRCUIT_H_
#define TACHYON_ZK_PLONK_EXAMPLES_MULTI_LOOKUP_CIRCUIT_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/constraint_system/circuit.h"

namespace tachyon::zk::plonk {

template <typename F>
class MultiLookupCircuitConfig {
 public:
  using Field = F;

  MultiLookupCircuitConfig() = default;
  MultiLookupCircuitConfig(const AdviceColumnKey& a, const AdviceColumnKey& b,
                           const AdviceColumnKey& c, const AdviceColumnKey& d,
                           const AdviceColumnKey& e, const FixedColumnKey& sa,
                           const FixedColumnKey& sb, const FixedColumnKey& sc,
                           const FixedColumnKey& sf, const FixedColumnKey& sm,
                           const FixedColumnKey& sp,
                           const LookupTableColumn& sl)
      : a_(a),
        b_(b),
        c_(c),
        d_(d),
        e_(e),
        sa_(sa),
        sb_(sb),
        sc_(sc),
        sf_(sf),
        sm_(sm),
        sp_(sp),
        sl_(sl) {}

  MultiLookupCircuitConfig Clone() const {
    return {a_, b_, c_, d_, e_, sa_, sb_, sc_, sf_, sm_, sp_, sl_};
  }

  const AdviceColumnKey& a() const { return a_; }
  const AdviceColumnKey& b() const { return b_; }
  const AdviceColumnKey& c() const { return c_; }
  const AdviceColumnKey& d() const { return d_; }
  const AdviceColumnKey& e() const { return e_; }
  const FixedColumnKey& sa() const { return sa_; }
  const FixedColumnKey& sb() const { return sb_; }
  const FixedColumnKey& sc() const { return sc_; }
  const FixedColumnKey& sf() const { return sf_; }
  const FixedColumnKey& sm() const { return sm_; }
  const FixedColumnKey& sp() const { return sp_; }
  const LookupTableColumn& sl() const { return sl_; }

  static MultiLookupCircuitConfig Configure(ConstraintSystem<F>& meta) {
    AdviceColumnKey e = meta.CreateAdviceColumn();
    AdviceColumnKey a = meta.CreateAdviceColumn();
    AdviceColumnKey b = meta.CreateAdviceColumn();
    FixedColumnKey sf = meta.CreateFixedColumn();
    AdviceColumnKey c = meta.CreateAdviceColumn();
    AdviceColumnKey d = meta.CreateAdviceColumn();
    InstanceColumnKey p = meta.CreateInstanceColumn();

    meta.EnableEquality(a);
    meta.EnableEquality(b);
    meta.EnableEquality(c);

    FixedColumnKey sm = meta.CreateFixedColumn();
    FixedColumnKey sa = meta.CreateFixedColumn();
    FixedColumnKey sb = meta.CreateFixedColumn();
    FixedColumnKey sc = meta.CreateFixedColumn();
    FixedColumnKey sp = meta.CreateFixedColumn();
    LookupTableColumn sl = meta.CreateLookupTableColumn();

    Selector dummy = meta.CreateComplexSelector();
    Selector dummy_2 = meta.CreateComplexSelector();
    Selector dummy_3 = meta.CreateComplexSelector();

    LookupTableColumn dummy_table = meta.CreateLookupTableColumn();

    //
    //   A         B      ...  sl
    // [
    //   instance  0      ...  0
    //   a         a      ...  0
    //   a         a²     ...  0
    //   a         a      ...  0
    //   a         a²     ...  0
    //   ...       ...    ...  ...
    //   ...       ...    ...  instance
    //   ...       ...    ...  a
    //   ...       ...    ...  a
    //   ...       ...    ...  0
    // ]
    //

    meta.Lookup("lookup", [&a, &sl](VirtualCells<F> meta) {
      std::unique_ptr<Expression<F>> a_expr = meta.QueryAny(a, Rotation::Cur());
      lookup::Pairs<std::unique_ptr<Expression<F>>, LookupTableColumn>
          lookup_pairs;
      lookup_pairs.emplace_back(std::move(a_expr), sl);
      return lookup_pairs;
    });

    meta.Lookup("lookup_same", [&a, &sl](VirtualCells<F> meta) {
      std::unique_ptr<Expression<F>> a_expr = meta.QueryAny(a, Rotation::Cur());
      lookup::Pairs<std::unique_ptr<Expression<F>>, LookupTableColumn>
          lookup_pairs;
      lookup_pairs.emplace_back(std::move(a_expr), sl);
      return lookup_pairs;
    });

    meta.Lookup("lookup_same", [&b, &dummy, &dummy_2, &dummy_3,
                                &dummy_table](VirtualCells<F> meta) {
      std::unique_ptr<Expression<F>> b_expr = meta.QueryAny(b, Rotation::Cur());
      std::unique_ptr<Expression<F>> dummy_expr = meta.QuerySelector(dummy);
      std::unique_ptr<Expression<F>> dummy_2_expr = meta.QuerySelector(dummy_2);
      std::unique_ptr<Expression<F>> dummy_3_expr = meta.QuerySelector(dummy_3);

      lookup::Pairs<std::unique_ptr<Expression<F>>, LookupTableColumn>
          lookup_pairs;
      lookup_pairs.emplace_back(std::move(dummy_expr) *
                                    std::move(dummy_2_expr) *
                                    std::move(dummy_3_expr) * std::move(b_expr),
                                dummy_table);
      return lookup_pairs;
    });

    meta.CreateGate("Combined add-mult", [&](VirtualCells<F> meta) {
      std::unique_ptr<Expression<F>> d_expr =
          meta.QueryAdvice(d, Rotation::Next());
      std::unique_ptr<Expression<F>> a_expr =
          meta.QueryAdvice(a, Rotation::Cur());
      std::unique_ptr<Expression<F>> sf_expr =
          meta.QueryFixed(sf, Rotation::Cur());
      std::unique_ptr<Expression<F>> e_expr =
          meta.QueryAdvice(e, Rotation::Prev());
      std::unique_ptr<Expression<F>> b_expr =
          meta.QueryAdvice(b, Rotation::Cur());
      std::unique_ptr<Expression<F>> c_expr =
          meta.QueryAdvice(c, Rotation::Cur());

      std::unique_ptr<Expression<F>> sa_expr =
          meta.QueryFixed(sa, Rotation::Cur());
      std::unique_ptr<Expression<F>> sb_expr =
          meta.QueryFixed(sb, Rotation::Cur());
      std::unique_ptr<Expression<F>> sc_expr =
          meta.QueryFixed(sc, Rotation::Cur());
      std::unique_ptr<Expression<F>> sm_expr =
          meta.QueryFixed(sm, Rotation::Cur());

      std::unique_ptr<Expression<F>> a_clone = a_expr.get()->Clone();
      std::unique_ptr<Expression<F>> b_clone = b_expr.get()->Clone();

      std::vector<Constraint<F>> constraints;
      constraints.emplace_back(
          std::move(a_clone) * std::move(sa_expr) +
          std::move(b_clone) * std::move(sb_expr) +
          std::move(a_expr) * std::move(b_expr) * std::move(sm_expr) -
          (std::move(c_expr) * std::move(sc_expr)) +
          std::move(sf_expr) * (std::move(d_expr) * std::move(e_expr)));

      return constraints;
    });

    meta.CreateGate("Public input", [&](VirtualCells<F> meta) {
      std::unique_ptr<Expression<F>> a_expr =
          meta.QueryAdvice(a, Rotation::Cur());
      std::unique_ptr<Expression<F>> p_expr =
          meta.QueryInstance(p, Rotation::Cur());
      std::unique_ptr<Expression<F>> sp_expr =
          meta.QueryFixed(sp, Rotation::Cur());
      std::vector<Constraint<F>> constraints;
      constraints.emplace_back(std::move(sp_expr) *
                               (std::move(a_expr) - std::move(p_expr)));
      return constraints;
    });

    meta.EnableEquality(sf);
    meta.EnableEquality(e);
    meta.EnableEquality(d);
    meta.EnableEquality(p);
    meta.EnableEquality(sm);
    meta.EnableEquality(sa);
    meta.EnableEquality(sb);
    meta.EnableEquality(sc);
    meta.EnableEquality(sp);

    return {a, b, c, d, e, sa, sb, sc, sf, sm, sp, sl};
  }

 private:
  AdviceColumnKey a_;
  AdviceColumnKey b_;
  AdviceColumnKey c_;
  AdviceColumnKey d_;
  AdviceColumnKey e_;
  FixedColumnKey sa_;
  FixedColumnKey sb_;
  FixedColumnKey sc_;
  FixedColumnKey sf_;
  FixedColumnKey sm_;
  FixedColumnKey sp_;
  LookupTableColumn sl_;
};

template <typename F>
class StandardPlonkChip {
 public:
  explicit StandardPlonkChip(MultiLookupCircuitConfig<F>&& config)
      : config_(std::move(config)) {}

  void PublicInput(Layouter<F>* layouter, const Value<F>& value) {
    layouter->AssignRegion("public_input", [this, &value](Region<F>& region) {
      region.AssignAdvice("value", config_.a(), 0,
                          [&value]() { return value; });
      region.AssignFixed("public", config_.sp(), 0,
                         []() { return Value<F>::Known(F::One()); });
    });
  }

  std::vector<AssignedCell<F>> RawMultiply(
      Layouter<F>* layouter, const std::vector<Value<F>>& values) {
    std::vector<AssignedCell<F>> ret;
    ret.reserve(3);

    layouter->AssignRegion(
        "raw_multiply", [this, &values, &ret](Region<F>& region) {
          AssignedCell<F> lhs = region.AssignAdvice(
              "lhs", config_.a(), 0, [&values]() { return values[0]; });
          ret.push_back(std::move(lhs));

          region.AssignAdvice("lhs⁴", config_.d(), 0,
                              [&values]() { return values[0].Pow(4); });

          AssignedCell<F> rhs = region.AssignAdvice(
              "rhs", config_.b(), 0, [&values]() { return values[1]; });
          ret.push_back(std::move(rhs));

          region.AssignAdvice("rhs⁴", config_.e(), 0,
                              [&values]() { return values[1].Pow(4); });

          AssignedCell<F> out = region.AssignAdvice(
              "out", config_.c(), 0, [&values]() { return values[2]; });
          ret.push_back(std::move(out));

          region.AssignFixed("a", config_.sa(), 0,
                             []() { return Value<F>::Known(F::Zero()); });
          region.AssignFixed("b", config_.sb(), 0,
                             []() { return Value<F>::Known(F::Zero()); });
          region.AssignFixed("c", config_.sc(), 0,
                             []() { return Value<F>::Known(F::One()); });
          region.AssignFixed("a * b", config_.sm(), 0,
                             []() { return Value<F>::Known(F::One()); });
        });

    return ret;
  }

  std::vector<AssignedCell<F>> RawAdd(Layouter<F>* layouter,
                                      const std::vector<Value<F>>& values) {
    std::vector<AssignedCell<F>> ret;
    ret.reserve(3);

    layouter->AssignRegion("raw_add", [this, &values, &ret](Region<F>& region) {
      AssignedCell<F> lhs = region.AssignAdvice(
          "lhs", config_.a(), 0, [&values]() { return values[0]; });
      ret.push_back(std::move(lhs));

      region.AssignAdvice("lhs⁴", config_.d(), 0,
                          [&values]() { return values[0].Pow(4); });

      AssignedCell<F> rhs = region.AssignAdvice(
          "rhs", config_.b(), 0, [&values]() { return values[1]; });
      ret.push_back(std::move(rhs));

      region.AssignAdvice("rhs⁴", config_.e(), 0,
                          [&values]() { return values[1].Pow(4); });

      AssignedCell<F> out = region.AssignAdvice(
          "out", config_.c(), 0, [&values]() { return values[2]; });
      ret.push_back(std::move(out));

      region.AssignFixed("a", config_.sa(), 0,
                         []() { return Value<F>::Known(F::One()); });
      region.AssignFixed("b", config_.sb(), 0,
                         []() { return Value<F>::Known(F::One()); });
      region.AssignFixed("c", config_.sc(), 0,
                         []() { return Value<F>::Known(F::One()); });
      region.AssignFixed("a * b", config_.sm(), 0,
                         []() { return Value<F>::Known(F::Zero()); });
    });

    return ret;
  }

  void Copy(Layouter<F>* layouter, const Cell& left, const Cell& right) {
    layouter->AssignRegion("copy", [&left, &right](Region<F>& region) {
      region.ConstrainEqual(left, right);
      region.ConstrainEqual(left, right);
    });
  }

  void LookupTableFromValues(Layouter<F>* layouter,
                             const std::vector<F>& values) {
    layouter->AssignLookupTable("", [this, &values](LookupTable<F>& table) {
      for (size_t i = 0; i < values.size(); ++i) {
        if (!table.AssignCell("table col", config_.sl(), i, [&values, i]() {
              return Value<F>::Known(values[i]);
            }))
          return false;
      }
      return true;
    });
  }

 private:
  MultiLookupCircuitConfig<F> config_;
};

template <typename F, template <typename> class _FloorPlanner>
class MultiLookupCircuit : public Circuit<MultiLookupCircuitConfig<F>> {
 public:
  using FloorPlanner = _FloorPlanner<MultiLookupCircuit<F, _FloorPlanner>>;

  MultiLookupCircuit() = default;
  MultiLookupCircuit(const F& a, const std::vector<F>& lookup_table)
      : a_(Value<F>::Known(a)), lookup_table_(lookup_table) {}

  std::unique_ptr<Circuit<MultiLookupCircuitConfig<F>>> WithoutWitness()
      const override {
    return std::make_unique<MultiLookupCircuit>();
  }

  static MultiLookupCircuitConfig<F> Configure(ConstraintSystem<F>& meta) {
    return MultiLookupCircuitConfig<F>::Configure(meta);
  }

  void Synthesize(MultiLookupCircuitConfig<F>&& config,
                  Layouter<F>* layouter) const override {
    StandardPlonkChip<F> cs(std::move(config));

    cs.PublicInput(layouter, Value<F>::Known(F::One() + F::One()));

    for (size_t i = 0; i < 10; ++i) {
      Value<F> a_squared = a_.SquareImpl();
      std::vector<Value<F>> mul_values;
      mul_values.reserve(3);
      mul_values.push_back(a_);
      mul_values.push_back(a_);
      mul_values.push_back(a_squared);
      std::vector<AssignedCell<F>> mul_cells =
          cs.RawMultiply(layouter, mul_values);

      Value<F> fin = a_squared.Add(a_);
      std::vector<Value<F>> add_values;
      add_values.reserve(3);
      add_values.push_back(a_);
      add_values.push_back(a_squared);
      add_values.push_back(fin);
      std::vector<AssignedCell<F>> add_cells = cs.RawAdd(layouter, add_values);

      cs.Copy(layouter, mul_cells[0].cell(), add_cells[0].cell());
      cs.Copy(layouter, add_cells[1].cell(), mul_cells[2].cell());
    }

    cs.LookupTableFromValues(layouter, lookup_table_);
  }

 private:
  Value<F> a_;
  std::vector<F> lookup_table_;
};
}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_MULTI_LOOKUP_CIRCUIT_H_
