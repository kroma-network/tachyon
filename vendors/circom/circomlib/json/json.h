#ifndef VENDORS_CIRCOM_CIRCOMLIB_JSON_JSON_H_
#define VENDORS_CIRCOM_CIRCOMLIB_JSON_JSON_H_

#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "circomlib/json/json_converter_forward.h"
#include "tachyon/base/files/file_util.h"

namespace tachyon::circom {

template <typename T>
bool WriteToJson(const T& value, const base::FilePath& path) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  ConvertToJson(value).Accept(writer);
  return base::WriteFile(path, buffer.GetString());
}

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_JSON_JSON_H_
