#include "tachyon/base/binding/cpp_constructor.h"

#include "tachyon/base/binding/cpp_type_names.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon::base {

// static
std::string_view CppConstructor::UnwrapPointer(std::string_view arg_type_name) {
  if (ConsumeSuffix(&arg_type_name, "*")) {
    return arg_type_name;
  } else if (ConsumePrefix(&arg_type_name, kCppSharedPtrTypePrefix) &&
             ConsumeSuffix(&arg_type_name, ">")) {
    return arg_type_name;
  } else if (ConsumePrefix(&arg_type_name, kCppUniquePtrTypePrefix) &&
             ConsumeSuffix(&arg_type_name, ">")) {
    return arg_type_name;
  }

  return arg_type_name;
}

CppConstructor::CppConstructor() = default;

CppConstructor::~CppConstructor() = default;

bool CppConstructor::HasDoublePointerArgument() const {
  for (std::string_view arg_type_name : arg_type_names_) {
    std::string_view unwrapped = UnwrapPointer(arg_type_name);
    if (unwrapped.length() == arg_type_name.length()) continue;
    std::string_view unwrapped2 = UnwrapPointer(unwrapped);
    if (unwrapped2.length() == unwrapped.length()) continue;
    return true;
  }
  return false;
}

void CppConstructor::ValidateAndMaybeDie() const {
  LOG_IF(DFATAL, HasDoublePointerArgument())
      << "You add the constructor that contains the double pointer ("
      << GetFunctionSignature(0) << ")";
}

std::string CppConstructor::GetFunctionSignature(
    size_t default_args_num) const {
  DCHECK_LE(default_args_num, arg_type_names_.size());

  size_t size = arg_type_names_.size() - default_args_num;

  if (size == 0) return "";

  std::stringstream ss;
  for (size_t i = 0; i < size - 1; ++i) {
    ss << arg_type_names_[i] << ",";
  }
  ss << arg_type_names_[size - 1];
  return ss.str();
}

}  // namespace tachyon::base
