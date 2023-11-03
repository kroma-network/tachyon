// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_CALCULATION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_CALCULATION_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/export.h"
#include "tachyon/zk/plonk/circuit/expressions/evaluator/value_source.h"
#include "tachyon/zk/plonk/circuit/expressions/evaluator/value_source_data.h"

namespace tachyon::zk {

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
  Calculation(const Calculation& other) { Copy(other); }
  Calculation(Calculation&& other) { Move(std::move(other)); }
  Calculation& operator=(const Calculation& other) {
    Copy(other);
    return *this;
  }
  Calculation& operator=(Calculation&& other) {
    Move(std::move(other));
    return *this;
  }
  ~Calculation() { Reset(); }

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
    if (type_ != other.type_) return false;
    switch (type_) {
      case Type::kAdd:
      case Type::kSub:
      case Type::kMul:
        return pair_ == other.pair_;
      case Type::kSquare:
      case Type::kDouble:
      case Type::kNegate:
      case Type::kStore:
        return value_ == other.value_;
      case Type::kHorner:
        return horner_ == other.horner_;
    }
    NOTREACHED();
    return false;
  }
  bool operator!=(const Calculation& other) const { return !operator==(other); }

  template <typename Poly, typename F = typename Poly::Field>
  F Evaluate(const ValueSourceData<Poly>& data) {
    switch (type_) {
      case Type::kAdd:
        return pair_.left.Get(data) + pair_.right.Get(data);
      case Type::kSub:
        return pair_.left.Get(data) - pair_.right.Get(data);
      case Type::kMul:
        return pair_.left.Get(data) * pair_.right.Get(data);
      case Type::kSquare:
        return value_.Get(data).Square();
      case Type::kDouble:
        return value_.Get(data).Double();
      case Type::kNegate:
        return -value_.Get(data);
      case Type::kStore:
        return value_.Get(data);
      case Type::kHorner: {
        F factor = horner_.factor.Get(data);
        F value = horner_.init.Get(data);
        for (const ValueSource& part : horner_.parts) {
          value = value * factor + part.Get(data);
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
      : type_(type), pair_(left, right) {}
  Calculation(Type type, const ValueSource& init,
              const std::vector<ValueSource>& parts, const ValueSource& factor)
      : type_(type), horner_(init, parts, factor) {}

  void Copy(const Calculation& other) {
    type_ = other.type_;
    switch (type_) {
      case Type::kAdd:
      case Type::kSub:
      case Type::kMul:
        pair_ = other.pair_;
        break;
      case Type::kSquare:
      case Type::kDouble:
      case Type::kNegate:
      case Type::kStore:
        value_ = other.value_;
        break;
      case Type::kHorner:
        horner_ = other.horner_;
        break;
    }
  }

  void Move(Calculation&& other) {
    type_ = other.type_;
    switch (type_) {
      case Type::kAdd:
      case Type::kSub:
      case Type::kMul:
        pair_ = other.pair_;
        break;
      case Type::kSquare:
      case Type::kDouble:
      case Type::kNegate:
      case Type::kStore:
        value_ = other.value_;
        break;
      case Type::kHorner:
        horner_ = std::move(other.horner_);
        break;
    }
  }

  void Reset() {
    if (type_ == Type::kHorner) {
      horner_.~HornerData();
    }
  }

  Type type_;
  union {
    ValueSource value_;
    Pair pair_;
    HornerData horner_;
  };
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_CALCULATION_H_
