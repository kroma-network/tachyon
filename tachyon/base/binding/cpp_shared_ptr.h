#ifndef TACHYON_BASE_BINDING_CPP_SHARED_PTR_H_
#define TACHYON_BASE_BINDING_CPP_SHARED_PTR_H_

#include <memory>
#include <type_traits>

#include "tachyon/base/binding/cpp_value.h"

namespace tachyon::base {

template <typename T>
class CppSharedPtr : public CppValue {
 public:
  explicit CppSharedPtr(const std::shared_ptr<T>& ptr) : ptr_(ptr) {}
  ~CppSharedPtr() override = default;

  bool IsCppSharedPtr() const override { return true; }

  void* raw_ptr() override {
    return reinterpret_cast<void*>(
        const_cast<std::remove_const_t<T>*>(ptr_.get()));
  }
  const void* raw_ptr() const override { return ptr_.get(); }

  bool is_const() const override { return std::is_const<T>::value; }

  std::shared_ptr<T> shared_ptr() { return ptr_; }

 private:
  std::shared_ptr<T> ptr_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_SHARED_PTR_H_
