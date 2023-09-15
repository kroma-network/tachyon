#include "tachyon/base/binding/test/move_only_int.h"

namespace tachyon::base::test {

MoveOnlyInt::MoveOnlyInt() : value_(0) {}

MoveOnlyInt::MoveOnlyInt(int value) : value_(value) {}

MoveOnlyInt::MoveOnlyInt(MoveOnlyInt&& other) : value_(other.value_) {
  other.value_ = 0;
}

MoveOnlyInt& MoveOnlyInt::operator=(MoveOnlyInt&& other) {
  value_ = other.value_;
  other.value_ = 0;
  return *this;
}

}  // namespace tachyon::base::test
