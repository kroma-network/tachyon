#ifndef TACHYON_NODE_MATH_BASE_BIG_INT_H_
#define TACHYON_NODE_MATH_BASE_BIG_INT_H_

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/math/base/big_int.h"
#include "tachyon/node/base/node_internals_forwards.h"

namespace tachyon::node {
namespace internal {

template <size_t N>
class CppValueTraits<tachyon::math::BigInt<N>> {
 public:
  static bool ToNativeValue(const Napi::Value& value,
                            tachyon::math::BigInt<N>* v) {
#if NAPI_VERSION > 5
    if (value.IsBigInt()) {
      int sign_bit;
      size_t word_count = N;
      value.As<Napi::BigInt>().ToWords(&sign_bit, &word_count, v->limbs);
      if (sign_bit == 1) return false;
      return word_count <= N;
    }
#endif
    if (!value.IsNumber()) return false;
    int64_t d = value.As<Napi::Number>().Int64Value();
    if (d < 0) return false;
    *v = tachyon::math::BigInt<N>(d);
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               const tachyon::math::BigInt<N>& value) {
#if NAPI_VERSION > 5
    return Napi::BigInt::New(info.Env(), false, N, value.limbs);
#else
    NAPI_THROW(Napi::TypeError::New(env, "BigInt is not supported"));
#endif
  }

  static std::string GetTypeName() {
    return absl::Substitute("tachyon::tachyon::math::Bigint<$0>", N);
  }
};

template <>
class CppValueTraits<tachyon::math::BigInt<1>> {
 public:
  static bool ToNativeValue(const Napi::Value& value,
                            tachyon::math::BigInt<1>* v) {
    uint64_t v_tmp;
    if (!CppValueTraits<uint64_t>::ToNativeValue(value, &v_tmp)) return false;
    *v = tachyon::math::BigInt<1>(v_tmp);
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               tachyon::math::BigInt<1>& value) {
    return CppValueTraits<uint64_t>::ToJSValue(info, value[0]);
  }

  static const char* GetTypeName() {
    return "tachyon::tachyon::math::Bigint<1>";
  }
};

}  // namespace internal
}  // namespace tachyon::node

#endif  // TACHYON_NODE_MATH_BASE_BIG_INT_H_
