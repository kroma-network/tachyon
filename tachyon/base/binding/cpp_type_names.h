#ifndef TACHYON_BASE_BINDING_CPP_TYPE_NAMES_H_
#define TACHYON_BASE_BINDING_CPP_TYPE_NAMES_H_

#include <string>
#include <string_view>
#include <vector>

#include "tachyon/export.h"

namespace tachyon::base {

constexpr const char* kCppBoolTypeName = "bool";
constexpr const char* kCppIntTypeName = "int";
constexpr const char* kCppUintTypeName = "uint";
constexpr const char* kCppInt64TypeName = "int64";
constexpr const char* kCppUint64TypeName = "uint64";
constexpr const char* kCppNumberTypeName = "number";
constexpr const char* kCppStringTypeName = "string";
constexpr const char* kCppVectorTypePrefix = "vector<";
constexpr const char* kCppMapTypePrefix = "map<";
constexpr const char* kCppTupleTypePrefix = "tuple<";
constexpr const char* kCppSharedPtrTypePrefix = "shared_ptr<";
constexpr const char* kCppUniquePtrTypePrefix = "unique_ptr<";

TACHYON_EXPORT std::string MakeCppVectorTypeName(std::string_view type);
TACHYON_EXPORT std::string MakeCppMapTypeName(std::string_view key_type,
                                              std::string_view value_type);
TACHYON_EXPORT std::string MakeCppTupleTypeName(
    const std::vector<std::string>& types);
TACHYON_EXPORT std::string MakeCppRawPtrTypeName(std::string_view type);
TACHYON_EXPORT std::string MakeCppSharedPtrTypeName(std::string_view type);
TACHYON_EXPORT std::string MakeCppUniquePtrTypeName(std::string_view type);

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_TYPE_NAMES_H_
