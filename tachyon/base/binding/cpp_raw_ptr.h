#ifndef TACHYON_BASE_BINDING_CPP_RAW_PTR_H_
#define TACHYON_BASE_BINDING_CPP_RAW_PTR_H_

#include <type_traits>

#include "tachyon/base/binding/cpp_value.h"

namespace tachyon::base {

template <typename T>
class CppRawPtr : public CppValue {
 public:
  explicit CppRawPtr(T* ptr) : ptr_(ptr) {}
  ~CppRawPtr() override = default;

  bool IsCppRawPtr() const override { return true; }

  void* raw_ptr() override {
    return reinterpret_cast<void*>(const_cast<std::remove_const_t<T>*>(ptr_));
  }
  const void* raw_ptr() const override { return ptr_; }

  bool is_const() const override { return std::is_const<T>::value; }

 private:
  T* ptr_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_RAW_PTR_H_
