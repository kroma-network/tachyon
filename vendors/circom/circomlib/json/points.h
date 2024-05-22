#ifndef VENDORS_CIRCOM_CIRCOMLIB_JSON_POINTS_H_
#define VENDORS_CIRCOM_CIRCOMLIB_JSON_POINTS_H_

#include <string_view>

#include "rapidjson/document.h"

#include "tachyon/math/elliptic_curves/affine_point.h"

namespace tachyon::circom::internal {

template <typename Curve>
void AddMember(rapidjson::Document& document, std::string_view member,
               const math::AffinePoint<Curve>& point) {
  using BaseField = typename math::AffinePoint<Curve>::BaseField;

  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  const BaseField& x = point.x();
  const BaseField& y = point.y();
  rapidjson::Value value;
  value.SetArray();
  if constexpr (BaseField::ExtensionDegree() == 1) {
    value.PushBack({x.ToString(), allocator}, allocator);
    value.PushBack({y.ToString(), allocator}, allocator);
    value.PushBack("1", allocator);
  } else {
    static_assert(BaseField::ExtensionDegree() == 2);
    {
      rapidjson::Value inner_array;
      inner_array.SetArray();
      inner_array.PushBack({x.c0().ToString(), allocator}, allocator);
      inner_array.PushBack({x.c1().ToString(), allocator}, allocator);
      value.PushBack(inner_array, allocator);
    }
    {
      rapidjson::Value inner_array;
      inner_array.SetArray();
      inner_array.PushBack({y.c0().ToString(), allocator}, allocator);
      inner_array.PushBack({y.c1().ToString(), allocator}, allocator);
      value.PushBack(inner_array, allocator);
    }
    {
      rapidjson::Value inner_array;
      inner_array.SetArray();
      inner_array.PushBack("1", allocator);
      inner_array.PushBack("0", allocator);
      value.PushBack(inner_array, allocator);
    }
  }
  document.AddMember(rapidjson::StringRef(member.data(), member.size()), value,
                     document.GetAllocator());
}

}  // namespace tachyon::circom::internal

#endif  // VENDORS_CIRCOM_CIRCOMLIB_JSON_POINTS_H_
