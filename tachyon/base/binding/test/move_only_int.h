#ifndef TACHYON_BASE_BINDING_TEST_MOVE_ONLY_INT_H_
#define TACHYON_BASE_BINDING_TEST_MOVE_ONLY_INT_H_

namespace tachyon::base::test {

class MoveOnlyInt {
 public:
  MoveOnlyInt();
  explicit MoveOnlyInt(int value);
  MoveOnlyInt(MoveOnlyInt&& other);
  MoveOnlyInt& operator=(MoveOnlyInt&& other);

  void set_value(int value) { value_ = value; }

  int value() const { return value_; }

 private:
  int value_;
};

}  // namespace tachyon::base::test

#endif  // TACHYON_BASE_BINDING_TEST_MOVE_ONLY_INT_H_
