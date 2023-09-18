#ifndef TACHYON_BASE_BINDING_CPP_TYPE_H_
#define TACHYON_BASE_BINDING_CPP_TYPE_H_

#include <string>

#include "tachyon/base/no_destructor.h"

namespace tachyon::base {

template <typename T>
class CppType {
 public:
  static CppType& Get() {
    static NoDestructor<CppType<T>> cpp_type;
    return *cpp_type;
  }

  void set_name(const std::string& name) { name_ = name; }

  const std::string& name() { return name_; }

  const void* unique_id() const {
    static int unique_address;
    return &unique_address;
  }

 private:
  friend class NoDestructor<CppType<T>>;

  CppType() = default;
  CppType(const CppType& other) = delete;
  CppType& operator=(const CppType& other) = delete;

  std::string name_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_TYPE_H_
