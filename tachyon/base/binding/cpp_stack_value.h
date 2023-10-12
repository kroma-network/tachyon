#ifndef TACHYON_BASE_BINDING_CPP_STACK_VALUE_H_
#define TACHYON_BASE_BINDING_CPP_STACK_VALUE_H_

#include <type_traits>
#include <utility>

#include "tachyon/base/binding/cpp_value.h"

namespace tachyon::base {

template <typename T>
class CppStackValue : public CppValue {
 public:
  explicit CppStackValue(const T& value) : value_(value) {}
  explicit CppStackValue(T&& value) : value_(std::move(value)) {}

  ~CppStackValue() override = default;

  bool IsCppStackValue() const override { return true; }

  void* raw_ptr() override {
    return reinterpret_cast<void*>(
        const_cast<std::remove_const_t<T>*>(&value_));
  }
  const void* raw_ptr() const override { return &value_; }

  bool is_const() const override { return std::is_const<T>::value; }

 private:
  T value_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_STACK_VALUE_H_
