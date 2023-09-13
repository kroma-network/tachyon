#ifndef TACHYON_BASE_BINDING_CPP_VALUE_FACTORY_H_
#define TACHYON_BASE_BINDING_CPP_VALUE_FACTORY_H_

#include <memory>

#include "tachyon/base/binding/cpp_raw_ptr.h"
#include "tachyon/base/binding/cpp_shared_ptr.h"
#include "tachyon/base/binding/cpp_stack_value.h"
#include "tachyon/base/binding/cpp_unique_ptr.h"

namespace tachyon::base {

template <typename T>
struct CppValueFactory {
  static std::unique_ptr<CppValue> Create(T value) {
    return std::make_unique<CppStackValue<T>>(std::move(value));
  }
};

template <typename T>
struct CppValueFactory<std::shared_ptr<T>> {
  static std::unique_ptr<CppValue> Create(std::shared_ptr<T> value) {
    return std::make_unique<CppSharedPtr<T>>(std::move(value));
  }
};

template <typename T, typename Deleter>
struct CppValueFactory<std::unique_ptr<T, Deleter>> {
  static std::unique_ptr<CppValue> Create(std::unique_ptr<T, Deleter> value) {
    return std::make_unique<CppUniquePtr<T, Deleter>>(std::move(value));
  }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_VALUE_FACTORY_H_
