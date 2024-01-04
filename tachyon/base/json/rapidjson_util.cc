#include "tachyon/base/json/rapidjson_util.h"

#include "absl/strings/substitute.h"

namespace tachyon::base {

namespace {

constexpr const char* kRapidJsonTypeNames[] = {
    "null", "false", "true", "object", "array", "string", "number"};

}  // namespace

std::string RapidJsonMismatchedTypeError(std::string_view key,
                                         std::string_view type,
                                         const rapidjson::Value& value) {
  return absl::Substitute("\"$0\" expects type \"$1\" but type \"$2\" comes",
                          key, type, kRapidJsonTypeNames[value.GetType()]);
}

}  // namespace tachyon::base
