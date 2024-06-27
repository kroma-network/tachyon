#ifndef TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_VARIABLE_H_
#define TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_VARIABLE_H_

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/export.h"

namespace tachyon::zk::air {

class TACHYON_EXPORT Variable {
 public:
  enum class Type {
    kPreprocessed,
    kMain,
    kPublic,
    kPermutation,
    kChallenge,
  };

  constexpr static Variable Preprocessed(size_t row_index, size_t col_index) {
    return Variable(Type::kPreprocessed, row_index, col_index);
  }
  constexpr static Variable Main(size_t row_index, size_t col_index) {
    return Variable(Type::kMain, row_index, col_index);
  }
  constexpr static Variable Public(size_t row_index) {
    return Variable(Type::kPublic, row_index, 0);
  }
  constexpr static Variable Permutation(size_t row_index,
                                        size_t col_index = 0) {
    return Variable(Type::kPermutation, row_index, col_index);
  }
  constexpr static Variable Challenge(size_t row_index, size_t col_index = 0) {
    return Variable(Type::kChallenge, row_index, col_index);
  }

  constexpr static std::string_view TypeToString(Type type) {
    switch (type) {
      case Type::kPreprocessed:
        return "Preprocessed";
      case Type::kMain:
        return "Main";
      case Type::kPublic:
        return "Public";
      case Type::kPermutation:
        return "Permutation";
      case Type::kChallenge:
        return "Challenge";
    }
    NOTREACHED();
    return "";
  }

  constexpr Type type() const { return type_; }
  constexpr size_t row_index() const { return row_index_; }
  constexpr size_t col_index() const { return col_index_; }

  size_t Degree() const {
    switch (type_) {
      case Type::kPreprocessed:
      case Type::kMain:
      case Type::kPublic:
        return 1;
      case Type::kPermutation:
      case Type::kChallenge:
        return 0;
    }
    NOTREACHED();
    return 0;
  }

  bool operator==(const Variable& other) const {
    return type_ == other.type_ && row_index_ == other.row_index_ &&
           col_index_ == other.col_index_;
  }

  bool operator!=(const Variable& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("type: $0, row_index: $1, col_index: $2",
                            TypeToString(type_), row_index_, col_index_);
  }

 private:
  constexpr Variable(Type type, size_t row_index, size_t col_index)
      : type_(type), row_index_(row_index), col_index_(col_index) {}

  Type type_;

  // NOTE(batzor): |row_index_| is index within current transition window. It is
  // not the row index of the whole trace.
  size_t row_index_;
  size_t col_index_;
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_VARIABLE_H_
