#ifndef TACHYON_NODE_BASE_NODE_INTERNALS_H_
#define TACHYON_NODE_BASE_NODE_INTERNALS_H_

#if defined(TACHYON_NODE_BINDING)

#include <string>
#include <string_view>
#include <vector>

#include "absl/container/inlined_vector.h"

#include "tachyon/base/binding/cpp_raw_ptr.h"
#include "tachyon/base/binding/cpp_shared_ptr.h"
#include "tachyon/base/binding/cpp_stack_value.h"
#include "tachyon/base/binding/cpp_type_names.h"
#include "tachyon/base/binding/cpp_unique_ptr.h"
#include "tachyon/base/numerics/safe_conversions.h"
#include "tachyon/base/template_util.h"
#include "tachyon/node/base/node_cpp_object.h"
#include "tachyon/node/base/node_internals_forwards.h"

namespace tachyon::node {
namespace internal {

template <typename T>
struct CppTypeWrapperInternal {
  using Type = base::CppStackValue<std::remove_reference_t<T>>;
};

template <typename T>
struct CppTypeWrapperInternal<T*> {
  using Type = T*;
};

template <typename T>
struct NativeCppTypeWrapper<
    T, std::enable_if_t<HasToNativeValue<std::decay_t<T>>::value>> {
  using Type = std::decay_t<T>;
};

template <typename T>
struct NativeCppTypeWrapper<
    T, std::enable_if_t<!HasToNativeValue<std::decay_t<T>>::value>> {
  using Type =
      typename CppTypeWrapperInternal<base::reference_to_pointer_t<T>>::Type;
};

template <typename T>
struct JSCppTypeWrapper<
    T, std::enable_if_t<HasToJSValue<std::decay_t<T>>::value>> {
  using Type = std::decay_t<T>;
};

template <typename T>
struct JSCppTypeWrapper<
    T, std::enable_if_t<!HasToJSValue<std::decay_t<T>>::value>> {
  using Type = typename CppTypeWrapperInternal<T>::Type;
};

template <>
class CppValueTraits<bool> {
 public:
  static bool ToNativeValue(const Napi::Value& value, bool* v) {
    if (!value.IsBoolean()) return false;
    *v = value.As<Napi::Boolean>().Value();
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, bool value) {
    return Napi::Boolean::New(info.Env(), value);
  }

  static const char* GetTypeName() { return base::kCppBoolTypeName; }
};

template <typename T>
class CppValueTraits<T, std::enable_if_t<std::is_integral<T>::value &&
                                         std::is_signed<T>::value &&
                                         sizeof(T) <= sizeof(int32_t)>> {
 public:
  static bool ToNativeValue(const Napi::Value& value, T* v) {
#if NAPI_VERSION > 5
    if (value.IsBigInt()) {
      int64_t d = value.As<Napi::BigInt>().Int64Value(nullptr);
      if (!base::IsValueInRangeForNumericType<T>(d)) return false;
      *v = d;
      return true;
    }
#endif
    if (!value.IsNumber()) return false;
    int32_t d = value.As<Napi::Number>().Int32Value();
    if (!base::IsValueInRangeForNumericType<T>(d)) return false;
    *v = d;
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, T value) {
    return Napi::Number::New(info.Env(), value);
  }

  static const char* GetTypeName() { return base::kCppIntTypeName; }
};

template <typename T>
class CppValueTraits<
    T, std::enable_if_t<
           std::is_integral<T>::value && !std::is_same<bool, T>::value &&
           !std::is_signed<T>::value && sizeof(T) <= sizeof(uint32_t)>> {
 public:
  static bool ToNativeValue(const Napi::Value& value, T* v) {
#if NAPI_VERSION > 5
    if (value.IsBigInt()) {
      uint64_t d = value.As<Napi::BigInt>().Uint64Value(nullptr);
      if (!base::IsValueInRangeForNumericType<T>(d)) return false;
      *v = d;
      return true;
    }
#endif
    if (!value.IsNumber()) return false;
    uint32_t d = value.As<Napi::Number>().Uint32Value();
    if (!base::IsValueInRangeForNumericType<T>(d)) return false;
    *v = d;
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, T value) {
    return Napi::Number::New(info.Env(), value);
  }

  static const char* GetTypeName() { return base::kCppUintTypeName; }
};

template <>
class CppValueTraits<int64_t> {
 public:
  static bool ToNativeValue(const Napi::Value& value, int64_t* v) {
#if NAPI_VERSION > 5
    if (value.IsBigInt()) {
      bool lossless;
      *v = value.As<Napi::BigInt>().Int64Value(&lossless);
      return lossless;
    }
#endif
    if (!value.IsNumber()) return false;
    *v = value.As<Napi::Number>().Int64Value();
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, int64_t value) {
#if NAPI_VERSION > 5
    return Napi::BigInt::New(info.Env(), value);
#else
    return Napi::Number::New(info.Env(), value);
#endif
  }

  static const char* GetTypeName() { return base::kCppInt64TypeName; }
};

template <>
class CppValueTraits<uint64_t> {
 public:
  static bool ToNativeValue(const Napi::Value& value, uint64_t* v) {
#if NAPI_VERSION > 5
    if (value.IsBigInt()) {
      bool lossless;
      *v = value.As<Napi::BigInt>().Uint64Value(&lossless);
      return lossless;
    }
#endif
    if (!value.IsNumber()) return false;
    int64_t d = value.As<Napi::Number>().Int64Value();
    if (d < 0) return false;
    *v = d;
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, uint64_t value) {
#if NAPI_VERSION > 5
    return Napi::BigInt::New(info.Env(), value);
#else
    return Napi::Number::New(info.Env(), value);
#endif
  }

  static const char* GetTypeName() { return base::kCppUint64TypeName; }
};

template <>
class CppValueTraits<float> {
 public:
  static bool ToNativeValue(const Napi::Value& value, float* v) {
    if (!value.IsNumber()) return false;
    *v = value.As<Napi::Number>().FloatValue();
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, float value) {
    return Napi::Number::New(info.Env(), value);
  }

  static const char* GetTypeName() { return base::kCppNumberTypeName; }
};

template <>
class CppValueTraits<double> {
 public:
  static bool ToNativeValue(const Napi::Value& value, double* v) {
    if (!value.IsNumber()) return false;
    *v = value.As<Napi::Number>().DoubleValue();
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, double value) {
    return Napi::Number::New(info.Env(), value);
  }

  static const char* GetTypeName() { return base::kCppNumberTypeName; }
};

template <>
class CppValueTraits<std::string> {
 public:
  static bool ToNativeValue(const Napi::Value& value, std::string* v) {
    if (!value.IsString()) return false;
    *v = value.As<Napi::String>().Utf8Value();
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               const std::string& value) {
    return Napi::String::New(info.Env(), value.c_str(), value.length());
  }

  static const char* GetTypeName() { return base::kCppStringTypeName; }
};

template <>
class CppValueTraits<std::string_view> {
 public:
  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               std::string_view value) {
    return Napi::String::New(info.Env(), value.data(), value.length());
  }
};

template <typename T>
class CppValueTraits<T, std::enable_if_t<std::is_enum<T>::value>> {
 public:
  using U = std::underlying_type_t<T>;

  static bool ToNativeValue(const Napi::Value& value, T* v) {
    return CppValueTraits<U>::ToNativeValue(value, reinterpret_cast<U*>(v));
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, T value) {
    return Napi::Number::New(info.Env(), static_cast<U>(value));
  }

  static const std::string& GetTypeName() {
    return base::CppType<T>::Get().name();
  }
};

template <typename Vector, typename value_type = typename Vector::value_type>
bool VectorToNativeValueHelper(const Napi::Value& value, Vector* v) {
  Vector values;
  Napi::Array arr = value.As<Napi::Array>();
  values.reserve(arr.Length());
  for (size_t i = 0; i < arr.Length(); ++i) {
    value_type value;
    if (!ToNativeValue(arr[i], &value)) return false;
    values.push_back(std::move(value));
  }
  *v = std::move(values);
  return true;
}

template <typename Vector, typename value_type = typename Vector::value_type>
Napi::Value VectorToJSValue(const Napi::CallbackInfo& info,
                            const Vector& value) {
  Napi::Array ret = Napi::Array::New(info.Env(), value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    ret[i] = CppValueTraits<value_type>::ToJSValue(info, value[i]);
  }
  return ret;
}

template <typename T>
class CppValueTraits<std::vector<T>> {
 public:
  static bool ToNativeValue(const Napi::Value& value, std::vector<T>* v) {
    return VectorToNativeValueHelper(value, v);
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               const std::vector<T>& value) {
    return VectorToJSValue(info, value);
  }

  static std::string GetTypeName() {
    return base::MakeCppVectorTypeName(
        CppValueTraits<NativeCppType<T>>::GetTypeName());
  }
};

template <typename T, size_t N>
class CppValueTraits<absl::InlinedVector<T, N>> {
 public:
  static bool ToNativeValue(const Napi::Value& value,
                            absl::InlinedVector<T, N>* v) {
    return VectorToNativeValueHelper(value, v);
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               const absl::InlinedVector<T, N>& value) {
    return VectorToJSValue(info, value);
  }

  static std::string GetTypeName() {
    return base::MakeCppVectorTypeName(
        CppValueTraits<NativeCppType<T>>::GetTypeName());
  }
};

template <typename T>
class CppValueTraits<T*> {
  using Class = std::decay_t<T>;
  using Object = NodeCppObject<Class>;

 public:
  static bool ToNativeValue(const Napi::Value& value, T** v) {
    if (!value.IsObject()) return false;
    if (!NodeConstructors::GetInstance().InstanceOf(
            value.ToObject(), base::CppType<T>::Get().name()))
      return false;
    Object* object = Napi::ObjectWrap<Object>::Unwrap(value.As<Napi::Object>());
    base::CppValue* cpp_value = object->value();
    if (!std::is_const<T>::value && cpp_value->is_const()) return false;
    T* raw_ptr = reinterpret_cast<T*>(cpp_value->raw_ptr());
    if (!raw_ptr) return false;
    *v = raw_ptr;
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, T* v) {
    return Object::NewInstance(info.Env(),
                               Napi::External<base::CppValue>::New(
                                   info.Env(), new base::CppRawPtr<T>(v)));
  }

  static std::string GetTypeName() {
    return base::MakeCppRawPtrTypeName(
        CppValueTraits<NativeCppType<T>>::GetTypeName());
  }
};

template <typename T>
class CppValueTraits<std::reference_wrapper<T>> {
  using Class = std::decay_t<T>;
  using Object = NodeCppObject<Class>;

 public:
  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               std::reference_wrapper<T> v) {
    return Object::NewInstance(
        info.Env(),
        Napi::External<base::CppValue>::New(
            info.Env(), new base::CppRawPtr<T>(std::addressof(v.get()))));
  }
};

template <typename T>
class CppValueTraits<std::shared_ptr<T>> {
  using Class = std::decay_t<T>;
  using Object = NodeCppObject<Class>;

 public:
  static bool ToNativeValue(const Napi::Value& value, std::shared_ptr<T>* v) {
    if (!value.IsObject()) return false;
    if (!NodeConstructors::GetInstance().InstanceOf(
            value.ToObject(), base::CppType<T>::Get().name()))
      return false;
    Object* object = Napi::ObjectWrap<Object>::Unwrap(value.As<Napi::Object>());
    base::CppValue* cpp_value = object->value();
    if (!std::is_const<T>::value && cpp_value->is_const()) return false;
    if (!cpp_value->IsCppSharedPtr()) return false;
    *v = cpp_value->ToCppSharedPtr<T>()->shared_ptr();
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               const std::shared_ptr<T>& v) {
    return Object::NewInstance(info.Env(),
                               Napi::External<base::CppValue>::New(
                                   info.Env(), new base::CppSharedPtr<T>(v)));
  }

  static std::string GetTypeName() {
    return base::MakeCppSharedPtrTypeName(
        CppValueTraits<NativeCppType<T>>::GetTypeName());
  }
};

template <typename T, typename Deleter>
class CppValueTraits<std::unique_ptr<T, Deleter>> {
  using Class = std::decay_t<T>;
  using Object = NodeCppObject<Class>;

 public:
  static bool ToNativeValue(const Napi::Value& value,
                            std::unique_ptr<T, Deleter>* v) {
    if (!value.IsObject()) return false;
    if (!NodeConstructors::GetInstance().InstanceOf(
            value.ToObject(), base::CppType<T>::Get().name()))
      return false;
    Object* object = Napi::ObjectWrap<Object>::Unwrap(value.As<Napi::Object>());
    base::CppValue* cpp_value = object->value();
    if (!std::is_const<T>::value && cpp_value->is_const()) return false;
    if (!cpp_value->IsCppUniquePtr()) return false;
    *v = std::move(cpp_value->ToCppUniquePtr<T, Deleter>()->unique_ptr());
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info,
                               std::unique_ptr<T, Deleter> v) {
    return Object::NewInstance(
        info.Env(),
        Napi::External<base::CppValue>::New(
            info.Env(), new base::CppUniquePtr<T, Deleter>(std::move(v))));
  }

  static std::string GetTypeName() {
    return base::MakeCppUniquePtrTypeName(
        CppValueTraits<NativeCppType<T>>::GetTypeName());
  }
};

template <typename T>
class CppValueTraits<base::CppStackValue<T>> {
  using Class = std::decay_t<T>;
  using Object = NodeCppObject<Class>;

 public:
  static bool ToNativeValue(const Napi::Value& value, T* v) {
    if (!value.IsObject()) return false;
    if (!NodeConstructors::GetInstance().InstanceOf(
            value.ToObject(), base::CppType<T>::Get().name()))
      return false;
    Object* object = Napi::ObjectWrap<Object>::Unwrap(value.As<Napi::Object>());
    base::CppValue* cpp_value = object->value();
    if (!std::is_const<T>::value && cpp_value->is_const()) return false;
    AssignHelper(cpp_value, v);
    return true;
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, const T& v) {
    return Object::NewInstance(info.Env(),
                               Napi::External<base::CppValue>::New(
                                   info.Env(), new base::CppStackValue<T>(v)));
  }

  static Napi::Value ToJSValue(const Napi::CallbackInfo& info, T&& v) {
    return Object::NewInstance(
        info.Env(), Napi::External<base::CppValue>::New(
                        info.Env(), new base::CppStackValue<T>(std::move(v))));
  }

  static const std::string& GetTypeName() {
    return base::CppType<T>::Get().name();
  }

 private:
  template <typename U = T,
            std::enable_if_t<std::is_copy_assignable<U>::value>* = nullptr>
  static void AssignHelper(base::CppValue* value, T* v) {
    *v = *reinterpret_cast<T*>(value->raw_ptr());
  }

  template <typename U = T,
            std::enable_if_t<!std::is_copy_assignable<U>::value>* = nullptr>
  static void AssignHelper(base::CppValue* value, T* v) {
    *v = std::move(*reinterpret_cast<T*>(value->raw_ptr()));
  }
};

template <typename T>
bool ToNativeValue(const Napi::Value& value, T* v) {
  return CppValueTraits<NativeCppType<T>>::ToNativeValue(value, v);
}

template <typename T>
Napi::Value ToJSValue(const Napi::CallbackInfo& info, T&& value) {
  return CppValueTraits<JSCppType<T>>::ToJSValue(info, std::forward<T>(value));
}

}  // namespace internal
}  // namespace tachyon::node

#endif  // defined(TACHYON_NODE_BINDING)

#endif  // TACHYON_NODE_BASE_NODE_INTERNALS_H_
