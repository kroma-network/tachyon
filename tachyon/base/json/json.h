#ifndef TACHYON_BASE_JSON_JSON_H_
#define TACHYON_BASE_JSON_JSON_H_

#include <string>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "tachyon/base/files/file_util.h"
#include "tachyon/base/json/rapidjson_util.h"

namespace tachyon::base {

// Parse json string and populate |value| on success.
template <typename T>
bool ParseJson(std::string_view content, T* value, std::string* error) {
  rapidjson::Document document;
  document.Parse(content.data(), content.length());
  if (document.HasParseError()) {
    *error =
        absl::Substitute("Failed to parse with error \"$0\" at offset $1",
                         rapidjson::GetParseError_En(document.GetParseError()),
                         document.GetErrorOffset());
    return false;
  }
  return RapidJsonValueConverter<T>::To(document.GetObject(), "", value, error);
}

// Load from file and parse json string and populate |value| on success.
template <typename T>
bool LoadAndParseJson(const FilePath& path, T* value, std::string* error) {
  std::string content;
  if (!ReadFileToString(path, &content)) {
    *error = absl::Substitute("Failed to read file: $0", path.value());
    return false;
  }
  return ParseJson(content, value, error);
}

// Write |value| to json string.
template <typename T>
std::string WriteToJson(const T& value) {
  rapidjson::Document document;
  rapidjson::Value json_value =
      RapidJsonValueConverter<T>::From(value, document.GetAllocator());
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  json_value.Accept(writer);
  return buffer.GetString();
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_JSON_JSON_H_
