// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_VARIABLE_H_
#define TACHYON_ZK_R1CS_VARIABLE_H_

#include <stddef.h>

#include <optional>
#include <string>

#include "tachyon/base/logging.h"
#include "tachyon/export.h"

namespace tachyon::zk::r1cs {

class TACHYON_EXPORT Variable {
 public:
  // NOTE(chokobole): THE ORDER OF ELEMENTS ARE IMPORTANT!! DO NOT CHANGE!
  enum class Type {
    // Represents the "zero" constant.
    kZero,
    // Represents the "one" constant.
    kOne,
    // Represents a public instance variable.
    kInstance,
    // Represents a private witness variable.
    kWitness,
    // Represents of a linear combination.
    kSymbolicLinearCombination,
  };

  static std::string_view TypeToString(Type type);

  Variable() = default;

  constexpr static Variable Zero() { return Variable(Type::kZero); }
  constexpr static Variable One() { return Variable(Type::kOne); }
  constexpr static Variable Instance(size_t index) {
    return Variable(Type::kInstance, index);
  }
  constexpr static Variable Witness(size_t index) {
    return Variable(Type::kWitness, index);
  }
  constexpr static Variable SymbolicLinearCombination(size_t index) {
    return Variable(Type::kSymbolicLinearCombination, index);
  }

  Type type() const { return type_; }
  size_t index() const { return index_; }

  std::optional<size_t> GetIndex(size_t witness_offset) const {
    switch (type_) {
      case Type::kOne:
        return 0;
      case Type::kInstance:
        return index_;
      case Type::kWitness:
        return witness_offset + index_;
      case Type::kZero:
      case Type::kSymbolicLinearCombination:
        return std::nullopt;
    }
    NOTREACHED();
    return std::nullopt;
  }
  std::optional<size_t> GetSymbolicLinearCombinationIndex() const {
    if (IsSymbolicLinearCombination()) return index_;
    return std::nullopt;
  }

  bool IsZero() const { return type_ == Type::kZero; }
  bool IsOne() const { return type_ == Type::kOne; }
  bool IsInstance() const { return type_ == Type::kInstance; }
  bool IsWitness() const { return type_ == Type::kWitness; }
  bool IsSymbolicLinearCombination() const {
    return type_ == Type::kSymbolicLinearCombination;
  }

  bool operator==(const Variable& other) const {
    return type_ == other.type_ && index_ == other.index_;
  }
  bool operator!=(const Variable& other) const { return !operator==(other); }
  bool operator<(const Variable& other) const {
    if (type_ == other.type_) {
      switch (type_) {
        case Type::kZero:
        case Type::kOne:
          return false;
        case Type::kInstance:
        case Type::kWitness:
        case Type::kSymbolicLinearCombination:
          return index_ < other.index_;
      }
      NOTREACHED();
    }
    return static_cast<int>(type_) < static_cast<int>(other.type_);
  }
  bool operator<=(const Variable& other) const {
    if (type_ == other.type_) {
      switch (type_) {
        case Type::kZero:
        case Type::kOne:
          return true;
        case Type::kInstance:
        case Type::kWitness:
        case Type::kSymbolicLinearCombination:
          return index_ <= other.index_;
      }
      NOTREACHED();
    }
    return static_cast<int>(type_) < static_cast<int>(other.type_);
  }
  bool operator>(const Variable& other) const { return !operator<=(other); }
  bool operator>=(const Variable& other) const { return !operator<(other); }

  std::string ToString() const;

 private:
  constexpr explicit Variable(Type type) : type_(type) {}
  constexpr Variable(Type type, size_t index) : type_(type), index_(index) {}

  Type type_ = Type::kZero;
  size_t index_ = 0;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_VARIABLE_H_
