#ifndef TACHYON_BASE_BINDING_CPP_UNIQUE_PTR_H_
#define TACHYON_BASE_BINDING_CPP_UNIQUE_PTR_H_

#include <memory>
#include <type_traits>

#include "tachyon/base/binding/cpp_value.h"

namespace tachyon::base {

template <typename T, typename Deleter>
class CppUniquePtr : public CppValue {
 public:
  explicit CppUniquePtr(std::unique_ptr<T, Deleter> ptr)
      : ptr_(std::move(ptr)) {}
  ~CppUniquePtr() override = default;

  bool IsCppUniquePtr() const override { return true; }

  void* raw_ptr() override {
    return reinterpret_cast<void*>(
        const_cast<std::remove_const_t<T>*>(ptr_.get()));
  }
  const void* raw_ptr() const override { return ptr_.get(); }

  bool is_const() const override { return std::is_const<T>::value; }

  std::unique_ptr<T, Deleter> unique_ptr() { return std::move(ptr_); }

 private:
  std::unique_ptr<T, Deleter> ptr_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_UNIQUE_PTR_H_
