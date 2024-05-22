#ifndef VENDORS_CIRCOM_CIRCOMLIB_JSON_PRIME_FIELD_H_
#define VENDORS_CIRCOM_CIRCOMLIB_JSON_PRIME_FIELD_H_

#include <type_traits>

#include "absl/types/span.h"

#include "circomlib/json/json_converter_forward.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::circom {

template <typename F>
class JsonSerializer<
    absl::Span<const F>,
    std::enable_if_t<std::is_base_of_v<tachyon::math::PrimeFieldBase<F>, F>>> {
 public:
  static rapidjson::Document ToJson(absl::Span<const F> prime_fields) {
    rapidjson::Document document;
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
    document.SetArray();

    for (const F& prime_field : prime_fields) {
      rapidjson::Value value;
      value.SetString(prime_field.ToString(), allocator);
      document.PushBack(value, allocator);
    }

    return document;
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_JSON_PRIME_FIELD_H_
