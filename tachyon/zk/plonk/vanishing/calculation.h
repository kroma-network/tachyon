// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_CALCULATION_H_
#define TACHYON_ZK_PLONK_VANISHING_CALCULATION_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/export.h"
#include "tachyon/zk/plonk/vanishing/evaluation_input.h"
#include "tachyon/zk/plonk/vanishing/value_source.h"

namespace tachyon::zk::plonk {

class TACHYON_EXPORT Calculation {
 public:
  enum class Type {
    kAdd,
    kSub,
    kMul,
    kSquare,
    kDouble,
    kNegate,
    kHorner,
    kStore,
  };

  Calculation() : Calculation(Type::kAdd) {}

  static Calculation Add(const ValueSource& left, const ValueSource& right) {
    return Calculation(Type::kAdd, left, right);
  }
  static Calculation Sub(const ValueSource& left, const ValueSource& right) {
    return Calculation(Type::kSub, left, right);
  }
  static Calculation Mul(const ValueSource& left, const ValueSource& right) {
    return Calculation(Type::kMul, left, right);
  }
  static Calculation Square(const ValueSource& value) {
    return Calculation(Type::kSquare, value);
  }
  static Calculation Double(const ValueSource& value) {
    return Calculation(Type::kDouble, value);
  }
  static Calculation Negate(const ValueSource& value) {
    return Calculation(Type::kNegate, value);
  }
  static Calculation Horner(const ValueSource& init,
                            std::vector<ValueSource> parts,
                            const ValueSource& factor) {
    return Calculation(Type::kHorner, init, std::move(parts), factor);
  }
  static Calculation Store(const ValueSource& value) {
    return Calculation(Type::kStore, value);
  }

  bool operator==(const Calculation& other) const {
    return type_ == other.type_ && pair_ == other.pair_ &&
           value_ == other.value_ && horner_ == other.horner_;
  }
  bool operator!=(const Calculation& other) const { return !operator==(other); }

  template <typename Evals, typename F>
  F Evaluate(const EvaluationInput<Evals>& data,
             const std::vector<F>& constants, const F& previous_value) const {
    switch (type_) {
      case Type::kAdd:
        return pair().left.Get(data, constants, previous_value) +
               pair().right.Get(data, constants, previous_value);
      case Type::kSub:
        return pair().left.Get(data, constants, previous_value) -
               pair().right.Get(data, constants, previous_value);
      case Type::kMul:
        return pair().left.Get(data, constants, previous_value) *
               pair().right.Get(data, constants, previous_value);
      case Type::kSquare:
        return value().Get(data, constants, previous_value).Square();
      case Type::kDouble:
        return value().Get(data, constants, previous_value).Double();
      case Type::kNegate:
        return -value().Get(data, constants, previous_value);
      case Type::kStore:
        return value().Get(data, constants, previous_value);
      case Type::kHorner: {
        const HornerData& honer = horner();
        F factor = honer.factor.Get(data, constants, previous_value);
        F value = honer.init.Get(data, constants, previous_value);
        for (const ValueSource& part : honer.parts) {
          value *= factor;
          value += part.Get(data, constants, previous_value);
        }
        return value;
      }
    }
    NOTREACHED();
    return F();
  }

  std::string ToString() const;

 private:
  struct Pair {
    ValueSource left;
    ValueSource right;

    Pair() = default;
    Pair(const ValueSource& left, const ValueSource& right)
        : left(left), right(right) {}

    bool operator==(const Pair& other) const {
      return left == other.left && right == other.right;
    }
    bool operator!=(const Pair& other) const { return !operator==(other); }
  };

  struct HornerData {
    ValueSource init;
    std::vector<ValueSource> parts;
    ValueSource factor;

    HornerData() = default;
    HornerData(const ValueSource& init, const std::vector<ValueSource>& parts,
               const ValueSource& factor)
        : init(init), parts(parts), factor(factor) {}

    bool operator==(const HornerData& other) const {
      return init == other.init && parts == other.parts &&
             factor == other.factor;
    }
    bool operator!=(const HornerData& other) const {
      return !operator==(other);
    }
  };

  explicit Calculation(Type type) : type_(type) {}
  Calculation(Type type, const ValueSource& value)
      : type_(type), value_(value) {}
  Calculation(Type type, const ValueSource& left, const ValueSource& right)
      : type_(type), pair_(Pair(left, right)) {}
  Calculation(Type type, const ValueSource& init,
              const std::vector<ValueSource>& parts, const ValueSource& factor)
      : type_(type), horner_(HornerData(init, parts, factor)) {}

  const ValueSource& value() const {
    CHECK(value_.has_value());
    return value_.value();
  }
  const Pair& pair() const {
    CHECK(pair_.has_value());
    return pair_.value();
  }
  const HornerData& horner() const {
    CHECK(horner_.has_value());
    return horner_.value();
  }

  Type type_;
  std::optional<ValueSource> value_;
  std::optional<Pair> pair_;
  std::optional<HornerData> horner_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_CALCULATION_H_
