// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_VALUE_H_
#define TACHYON_ZK_VALUE_H_

#include <optional>
#include <string>
#include <utility>

#include "tachyon/math/base/field.h"

namespace tachyon::zk {

template <typename T>
class Value : public math::Field<Value<T>> {
 public:
  constexpr static Value Unknown() { return Value(); }

  constexpr static Value Known(const T& value) { return Value(value); }

  constexpr static Value Known(T&& value) { return Value(std::move(value)); }

  constexpr static Value Zero() { return Value(T::Zero()); }

  constexpr static Value One() { return Value(T::One()); }

  static Value Random() { return Value(T::Random()); }

  constexpr bool IsZero() const {
    CHECK(!IsNone());
    return value_->IsZero();
  }

  constexpr bool IsOne() const {
    CHECK(!IsNone());
    return value_->IsOne();
  }

  constexpr bool IsNone() const { return !value_.has_value(); }

  std::string ToString() const {
    if (IsNone()) return "None";
    return value_->ToString();
  }

  constexpr bool operator==(const Value& other) const {
    if (IsNone()) return other.IsNone();
    if (other.IsNone()) return false;
    return *value_ == *other.value_;
  }
  constexpr bool operator!=(const Value& other) const {
    return operator!=(other);
  }

  const T* operator->() const { return value_.operator->(); }
  T* operator->() { return value_.operator->(); }

  // AdditiveSemigroup methods
  constexpr Value& AddInPlace(const Value& other) {
    CHECK(!other.IsNone());
    return AddInPlace(*other.value_);
  }

  constexpr Value& AddInPlace(const T& other) {
    CHECK(!IsNone());
    *value_ += other;
    return *this;
  }

  constexpr Value Add(const Value& other) const {
    CHECK(!other.IsNone());
    return Add(*other.value_);
  }

  constexpr Value Add(const T& other) const {
    CHECK(!IsNone());
    return Value::Known(*value_ + other);
  }

  constexpr Value& DoubleInPlace() {
    CHECK(!IsNone());
    value_->DoubleInPlace();
    return *this;
  }

  // AdditiveGroup methods
  constexpr Value& SubInPlace(const Value& other) {
    CHECK(!other.IsNone());
    return SubInPlace(*other.value_);
  }

  constexpr Value& SubInPlace(const T& other) {
    CHECK(!IsNone());
    *value_ -= other;
    return *this;
  }

  constexpr Value Sub(const Value& other) const {
    CHECK(!other.IsNone());
    return Sub(*other.value_);
  }

  constexpr Value Sub(const T& other) const {
    CHECK(!IsNone());
    return Value::Known(*value_ - other);
  }

  constexpr Value& NegInPlace() {
    CHECK(!IsNone());
    value_->NegInPlace();
    return *this;
  }

  // MultiplicativeSemigroup methods
  constexpr Value& MulInPlace(const Value& other) {
    CHECK(!other.IsNone());
    return MulInPlace(*other.value_);
  }

  constexpr Value& MulInPlace(const T& other) {
    CHECK(!IsNone());
    *value_ *= other;
    return *this;
  }

  constexpr Value Mul(const Value& other) const {
    CHECK(!other.IsNone());
    return Mul(*other.value_);
  }

  constexpr Value Mul(const T& other) const {
    CHECK(!IsNone());
    return Value::Known(*value_ * other);
  }

  constexpr Value& SquareInPlace() {
    CHECK(!IsNone());
    value_->SquareInPlace();
    return *this;
  }

  // MultiplicativeGroup methods
  constexpr Value& DivInPlace(const Value& other) {
    CHECK(!other.IsNone());
    return DivInPlace(*other.value_);
  }

  constexpr Value& DivInPlace(const T& other) {
    CHECK(!IsNone());
    *value_ /= other;
    return *this;
  }

  constexpr Value Div(const Value& other) const {
    CHECK(!other.IsNone());
    return Div(*other.value_);
  }

  constexpr Value Div(const T& other) const {
    CHECK(!IsNone());
    return Value::Known(*value_ / other);
  }

  constexpr Value& InverseInPlace() {
    CHECK(!IsNone());
    value_->InverseInPlace();
    return *this;
  }

 private:
  constexpr Value() = default;
  constexpr explicit Value(const T& value) : value_(value) {}
  constexpr explicit Value(T&& value) : value_(std::move(value)) {}

  std::optional<T> value_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_VALUE_H_
