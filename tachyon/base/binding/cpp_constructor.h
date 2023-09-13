#ifndef TACHYON_BASE_BINDING_CPP_CONSTRUCTOR_H_
#define TACHYON_BASE_BINDING_CPP_CONSTRUCTOR_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tachyon/export.h"

namespace tachyon::base {

class TACHYON_EXPORT CppConstructor {
 public:
  CppConstructor();
  CppConstructor(const CppConstructor& other) = delete;
  CppConstructor& operator=(const CppConstructor& other) = delete;
  ~CppConstructor();

  bool HasDoublePointerArgument() const;
  void ValidateAndMaybeDie() const;

  size_t GetArgsNum() const { return arg_type_names_.size(); }
  size_t GetDefaultArgsNum() const { return default_args_num_; }
  void SetDefaultArgsNum(size_t default_args_num) {
    default_args_num_ = default_args_num;
  }
  std::string GetFunctionSignature(size_t default_args_num) const;

 protected:
  static std::string_view UnwrapPointer(std::string_view arg_type_name);

  size_t default_args_num_ = 0;
  std::vector<std::string> arg_type_names_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_CONSTRUCTOR_H_
