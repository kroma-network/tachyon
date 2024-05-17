#ifndef VENDORS_CIRCOM_CIRCOMLIB_JSON_JSON_CONVERTER_FORWARD_H_
#define VENDORS_CIRCOM_CIRCOMLIB_JSON_JSON_CONVERTER_FORWARD_H_

#include "rapidjson/document.h"

namespace tachyon::circom {

template <typename T, typename SFINAE = void>
class JsonSerializer;

template <typename T>
rapidjson::Document ConvertToJson(const T& value) {
  return JsonSerializer<T>::ToJson(value);
}

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_JSON_JSON_CONVERTER_FORWARD_H_
