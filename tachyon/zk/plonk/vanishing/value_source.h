// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VALUE_SOURCE_H_
#define TACHYON_ZK_PLONK_VANISHING_VALUE_SOURCE_H_

#include <stddef.h>

#include <string>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/export.h"
#include "tachyon/zk/plonk/vanishing/evaluation_input.h"

namespace tachyon::zk::plonk {

class TACHYON_EXPORT ValueSource {
 public:
  enum class Type {
    kConstant,
    kIntermediate,
    kFixed,
    kAdvice,
    kInstance,
    kChallenge,
    kBeta,
    kGamma,
    kTheta,
    kY,
    kPreviousValue,
  };

  ValueSource() : ValueSource(Type::kConstant, 0) {}

  // NOTE(chokobole): |GraphEvaluator| is already initialized with 3 constant
  // elements. See graph_evaluator.h
  static ValueSource ZeroConstant() { return ValueSource::Constant(0); }
  static ValueSource OneConstant() { return ValueSource::Constant(1); }
  static ValueSource TwoConstant() { return ValueSource::Constant(2); }

  static ValueSource Constant(size_t index) {
    return ValueSource(Type::kConstant, index);
  }
  static ValueSource Intermediate(size_t index) {
    return ValueSource(Type::kIntermediate, index);
  }
  static ValueSource Fixed(size_t column_index, size_t rotation_index) {
    return ValueSource(Type::kFixed, column_index, rotation_index);
  }
  static ValueSource Advice(size_t column_index, size_t rotation_index) {
    return ValueSource(Type::kAdvice, column_index, rotation_index);
  }
  static ValueSource Instance(size_t column_index, size_t rotation_index) {
    return ValueSource(Type::kInstance, column_index, rotation_index);
  }
  static ValueSource Challenge(size_t index) {
    return ValueSource(Type::kChallenge, index);
  }
  static ValueSource Beta() { return ValueSource(Type::kBeta); }
  static ValueSource Gamma() { return ValueSource(Type::kGamma); }
  static ValueSource Theta() { return ValueSource(Type::kTheta); }
  static ValueSource Y() { return ValueSource(Type::kY); }
  static ValueSource PreviousValue() {
    return ValueSource(Type::kPreviousValue);
  }

  Type type() const { return type_; }
  size_t index() const { return index_; }
  size_t column_index() const { return column_index_; }
  size_t rotation_index() const { return rotation_index_; }

  bool IsZeroConstant() const {
    return type_ == Type::kConstant && index_ == 0;
  }
  bool IsOneConstant() const { return type_ == Type::kConstant && index_ == 1; }
  bool IsTwoConstant() const { return type_ == Type::kConstant && index_ == 2; }

  // NOTE(chokobole): I am not sure why this is needed. I implemented == and <=
  // only because I believe this is the only operator used. See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/evaluation.rs#L36-L37
  // and in |GraphEvaluator<F>::Evaluate()| in graph_evaluator.h.
  bool operator==(const ValueSource& other) const {
    if (type_ != other.type_) return false;
    switch (type_) {
      case Type::kConstant:
      case Type::kIntermediate:
      case Type::kChallenge:
        return index_ == other.index_;
      case Type::kFixed:
      case Type::kAdvice:
      case Type::kInstance:
        return column_index_ == other.column_index_ &&
               rotation_index_ == other.rotation_index_;
      case Type::kBeta:
      case Type::kGamma:
      case Type::kTheta:
      case Type::kY:
      case Type::kPreviousValue:
        return true;
    }
    NOTREACHED();
    return false;
  }

  bool operator<=(const ValueSource& other) const {
    if (type_ == other.type_) {
      switch (type_) {
        case Type::kConstant:
        case Type::kIntermediate:
        case Type::kChallenge:
          return index_ <= other.index_;
        case Type::kFixed:
        case Type::kAdvice:
        case Type::kInstance:
          if (column_index_ == other.column_index_) {
            return rotation_index_ <= other.rotation_index_;
          }
          return column_index_ < other.column_index_;
        case Type::kBeta:
        case Type::kGamma:
        case Type::kTheta:
        case Type::kY:
        case Type::kPreviousValue:
          return true;
      }
      NOTREACHED();
      return false;
    }
    return type_ < other.type_;
  }

  template <typename Evals, typename F>
  F Get(const EvaluationInput<Evals>& data, const std::vector<F>& constants,
        const F& previous_value) const {
    switch (type_) {
      case Type::kConstant:
        return constants[index_];
      case Type::kIntermediate:
        return data.intermediates()[index_];
      case Type::kChallenge:
        return data.challenges()[index_];
      case Type::kFixed:
        return data.table()
            .GetFixedColumns()[column_index_]
                              [data.rotations()[rotation_index_]];
      case Type::kAdvice:
        return data.table()
            .GetAdviceColumns()[column_index_]
                               [data.rotations()[rotation_index_]];
      case Type::kInstance:
        return data.table()
            .GetInstanceColumns()[column_index_]
                                 [data.rotations()[rotation_index_]];
      case Type::kBeta:
        return data.beta();
      case Type::kGamma:
        return data.gamma();
      case Type::kTheta:
        return data.theta();
      case Type::kY:
        return data.y();
      case Type::kPreviousValue:
        return previous_value;
    }
    NOTREACHED();
    return F();
  }

  std::string ToString() const;

 private:
  explicit ValueSource(Type type) : type_(type) {}
  ValueSource(Type type, size_t index) : type_(type), index_(index) {}
  ValueSource(Type type, size_t column_index, size_t rotation_index)
      : type_(type),
        column_index_(column_index),
        rotation_index_(rotation_index) {}

  Type type_;
  union {
    size_t index_;
    struct {
      size_t column_index_;
      size_t rotation_index_;
    };
  };
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_VALUE_SOURCE_H_
