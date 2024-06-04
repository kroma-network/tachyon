#include "tachyon/base/binding/cpp_type_names.h"

#include "absl/strings/substitute.h"

namespace tachyon::base {

std::string MakeCppVectorTypeName(std::string_view type) {
  return absl::Substitute("$0$1>", kCppVectorTypePrefix, type);
}

std::string MakeCppOptionalTypeName(std::string_view type) {
  return absl::Substitute("$0$1>", kCppOptionalPrefix, type);
}

std::string MakeCppMapTypeName(std::string_view key_type,
                               std::string_view value_type) {
  return absl::Substitute("$0$1,$2>", kCppMapTypePrefix, key_type, value_type);
}

std::string MakeCppTupleTypeName(const std::vector<std::string>& types) {
  return absl::Substitute("$0$1>", kCppTupleTypePrefix,
                          absl::StrJoin(types, ","));
}

std::string MakeCppRawPtrTypeName(std::string_view type) {
  return absl::Substitute("$0*", type);
}

std::string MakeCppSharedPtrTypeName(std::string_view type) {
  return absl::Substitute("$0$1>", kCppSharedPtrTypePrefix, type);
}

std::string MakeCppUniquePtrTypeName(std::string_view type) {
  return absl::Substitute("$0$1>", kCppUniquePtrTypePrefix, type);
}

}  // namespace tachyon::base
