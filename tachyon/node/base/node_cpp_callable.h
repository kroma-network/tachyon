#ifndef TACHYON_NODE_BASE_NODE_CPP_CALLABLE_H_
#define TACHYON_NODE_BASE_NODE_CPP_CALLABLE_H_

#include <tuple>

#include "absl/strings/substitute.h"
#include "third_party/node_addon_api/napi.h"

#include "tachyon/base/binding/callable_util.h"
#include "tachyon/base/binding/cpp_value.h"
#include "tachyon/base/binding/property_util.h"
#include "tachyon/base/functional/functor_traits.h"
#include "tachyon/node/base/node_cpp_callable_data.h"
#include "tachyon/node/base/node_errors.h"
#include "tachyon/node/base/node_internals_forwards.h"

namespace tachyon::node {

namespace internal {

inline bool CallbackInfoToTuple(const Napi::CallbackInfo& info, size_t nargs,
                                std::tuple<>* v, std::make_index_sequence<0>) {
  return true;
}

template <typename Tuple, size_t N>
bool CallbackInfoToTuple(const Napi::CallbackInfo& info, size_t nargs, Tuple* v,
                         std::index_sequence<N>) {
  if (N >= nargs) return true;
  if (!ToNativeValue(info[N], &std::get<N>(*v))) {
    NAPI_THROW(InvalidArgument(info.Env(), N), false);
  }
  return true;
}

template <typename Tuple, size_t N, size_t... Is>
bool CallbackInfoToTuple(const Napi::CallbackInfo& info, size_t nargs, Tuple* v,
                         std::index_sequence<N, N + 1, Is...>) {
  if (!CallbackInfoToTuple(info, nargs, v, std::index_sequence<N>{}))
    return false;
  return CallbackInfoToTuple(info, nargs, v,
                             std::index_sequence<N + 1, Is...>{});
}

template <typename Tuple>
bool CallbackInfoToTuple(const Napi::CallbackInfo& info, size_t nargs,
                         Tuple* v) {
  return CallbackInfoToTuple(
      info, nargs, v,
      std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

inline void SetDefaultArguments(std::tuple<>, std::make_index_sequence<0>,
                                const std::vector<void*>&, size_t&) {}

template <typename Tuple, size_t Idx>
void SetDefaultArguments(Tuple& args, std::index_sequence<Idx>,
                         const std::vector<void*>& default_args,
                         size_t& required_default_args) {
  using T = std::tuple_element_t<Idx, Tuple>;
  constexpr size_t N = std::tuple_size<Tuple>::value;
  if (Idx < N - required_default_args) return;
  std::get<Idx>(args) = std::move(*reinterpret_cast<T*>(
      default_args[default_args.size() - required_default_args]));
  --required_default_args;
}

template <typename Tuple, size_t Idx, size_t... Is>
void SetDefaultArguments(Tuple& args, std::index_sequence<Idx, Idx + 1, Is...>,
                         const std::vector<void*>& default_args,
                         size_t& required_default_args) {
  SetDefaultArguments(args, std::index_sequence<Idx>{}, default_args,
                      required_default_args);
  SetDefaultArguments(args, std::index_sequence<Idx + 1, Is...>{}, default_args,
                      required_default_args);
}

template <typename Tuple>
void SetDefaultArguments(Tuple& args, const std::vector<void*>& default_args,
                         size_t& required_default_args) {
  SetDefaultArguments(args,
                      std::make_index_sequence<std::tuple_size<Tuple>::value>{},
                      default_args, required_default_args);
}

}  // namespace internal

class NodeCppCallable {
  template <typename ArgList>
  using ConvertTypeListToDeclarableTuple =
      base::internal::ConvertTypeListToDeclarableTuple<
          internal::HasToNativeValue, ArgList>;

#define PRE_CALL(info, data, ...)                                  \
  using Tuple = ConvertTypeListToDeclarableTuple<ArgList>;         \
  Tuple args;                                                      \
  size_t nargs = info.Length();                                    \
  constexpr size_t N = std::tuple_size<Tuple>::value;              \
  if (nargs > N) {                                                 \
    NAPI_THROW(WrongNumberOfArguments(info.Env()), __VA_ARGS__);   \
  } else if (nargs < N) {                                          \
    size_t required_default_args = N - nargs;                      \
    if (data->default_args.size() < required_default_args) {       \
      NAPI_THROW(WrongNumberOfArguments(info.Env()), __VA_ARGS__); \
    }                                                              \
    internal::SetDefaultArguments(args, data->default_args,        \
                                  required_default_args);          \
  }                                                                \
  internal::CallbackInfoToTuple(info, nargs, &args);               \
  if (info.Env().IsExceptionPending()) {                           \
    return __VA_ARGS__;                                            \
  }

#define arg_cast base::internal::arg_cast
#define ret_cast base::internal::RetTypeCaster<internal::HasToJSValue>::cast
#define property_cast \
  base::internal::PropertyTypeCaster<internal::HasToJSValue>::cast

 public:
  template <typename Functor,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::ExtractArgs<RunType>,
            std::enable_if_t<!std::is_same<ReturnType, void>::value>* = nullptr>
  static Napi::Value Call(const Napi::CallbackInfo& info) {
    NodeCppFunctionData* data =
        reinterpret_cast<NodeCppFunctionData*>(info.Data());
    PRE_CALL(info, data, info.Env().Null())
    Functor f = reinterpret_cast<Functor>(data->function);
    return internal::ToJSValue(info, ret_cast<ReturnType>(CallHelper(f, args)));
  }

  template <typename Functor,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::ExtractArgs<RunType>,
            std::enable_if_t<std::is_same<ReturnType, void>::value>* = nullptr>
  static void Call(const Napi::CallbackInfo& info) {
    NodeCppFunctionData* data =
        reinterpret_cast<NodeCppFunctionData*>(info.Data());
    PRE_CALL(info, data)
    Functor f = reinterpret_cast<Functor>(data->function);
    CallHelper(f, args);
  }

  template <typename Functor,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::ExtractArgs<RunType>>
  static std::unique_ptr<base::CppValue> CallConstructor(
      const Napi::CallbackInfo& info, NodeCppConstructorData* data) {
    PRE_CALL(info, data, nullptr);
    Functor f = reinterpret_cast<Functor>(data->cpp_function);
    return CallHelper(f, args);
  }

  template <typename Functor, typename Class,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::DropTypeListItem<
                1, base::internal::ExtractArgs<RunType>>,
            std::enable_if_t<!std::is_same<ReturnType, void>::value>* = nullptr>
  static Napi::Value CallMethod(const Napi::CallbackInfo& info, Class* cls) {
    NodeCppMethodData<Functor>* data =
        reinterpret_cast<NodeCppMethodData<Functor>*>(info.Data());
    if (!data->is_const && std::is_const<Class>::value) {
      NAPI_THROW(CallNonConstMethod(info.Env()), info.Env().Null());
    }
    PRE_CALL(info, data, info.Env().Null())
    Functor* f = data->method;
    return internal::ToJSValue(info,
                               ret_cast<ReturnType>(CallHelper(*f, cls, args)));
  }

  template <typename Functor, typename Class,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::DropTypeListItem<
                1, base::internal::ExtractArgs<RunType>>,
            std::enable_if_t<std::is_same<ReturnType, void>::value>* = nullptr>
  static void CallMethod(const Napi::CallbackInfo& info, Class* cls) {
    NodeCppMethodData<Functor>* data =
        reinterpret_cast<NodeCppMethodData<Functor>*>(info.Data());
    PRE_CALL(info, data)
    Functor* f = data->method;
    CallHelper(*f, cls, args);
  }

  template <typename Getter,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Getter>,
            typename ReturnType = typename FunctorTraits::ReturnType>
  static Napi::Value CallStaticGetter(const Napi::CallbackInfo& info) {
    NodeCppStaticPropertyAccessorData* accessor =
        reinterpret_cast<NodeCppStaticPropertyAccessorData*>(info.Data());
    Getter f = reinterpret_cast<Getter>(accessor->getter);
    return internal::ToJSValue(
        info, ret_cast<ReturnType>(CallHelper(*f, std::tuple<>{})));
  }

  template <typename Setter,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Setter>,
            typename RunType = typename FunctorTraits::RunType,
            typename ArgList = base::internal::ExtractArgs<RunType>>
  static void CallStaticSetter(const Napi::CallbackInfo& info,
                               const Napi::Value& value) {
    using Tuple = ConvertTypeListToDeclarableTuple<ArgList>;
    Tuple args;
    if (!internal::ToNativeValue(value, &std::get<0>(args))) {
      NAPI_THROW(InvalidArgument0(info.Env()));
    }
    NodeCppStaticPropertyAccessorData* accessor =
        reinterpret_cast<NodeCppStaticPropertyAccessorData*>(info.Data());
    Setter f = reinterpret_cast<Setter>(accessor->setter);
    CallHelper(*f, args);
  }

  template <typename Getter, typename Setter, typename Class,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Getter>,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename PropertyAccessorData =
                NodeCppPropertyAccessorData<Getter, Setter>>
  static Napi::Value CallGetter(const Napi::CallbackInfo& info, Class* cls) {
    PropertyAccessorData* accessor =
        reinterpret_cast<PropertyAccessorData*>(info.Data());
    Getter* f = reinterpret_cast<Getter*>(accessor->getter);
    return internal::ToJSValue(
        info, ret_cast<ReturnType>(CallHelper(*f, cls, std::tuple<>{})));
  }

  template <typename Getter, typename Setter, typename Class,
            typename PropertyAccessorData =
                NodeCppPropertyAccessorData<Getter, Setter>,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Setter>,
            typename RunType = typename FunctorTraits::RunType,
            typename ArgList = base::internal::DropTypeListItem<
                1, base::internal::ExtractArgs<RunType>>>
  static void CallSetter(const Napi::CallbackInfo& info, Class* cls,
                         const Napi::Value& value) {
    using Tuple = ConvertTypeListToDeclarableTuple<ArgList>;
    Tuple args;
    if (!internal::ToNativeValue(value, &std::get<0>(args))) {
      NAPI_THROW(InvalidArgument0(info.Env()));
    }
    PropertyAccessorData* accessor =
        reinterpret_cast<PropertyAccessorData*>(info.Data());
    Setter* f = reinterpret_cast<Setter*>(accessor->setter);
    CallHelper(*f, cls, args);
  }

  template <typename T>
  static Napi::Value GetStaticProperty(const Napi::CallbackInfo& info) {
    T* ptr = reinterpret_cast<T*>(info.Data());
    return internal::ToJSValue(info, property_cast(*ptr));
  }

  template <typename T>
  static void SetStaticProperty(const Napi::CallbackInfo& info,
                                const Napi::Value& value) {
    using Tuple = ConvertTypeListToDeclarableTuple<base::internal::TypeList<T>>;
    Tuple v;
    if (!internal::ToNativeValue(value, &std::get<0>(v))) {
      NAPI_THROW(InvalidArgument0(info.Env()));
    }
    *reinterpret_cast<T*>(info.Data()) = std::move(std::get<0>(v));
  }

  template <typename Class, typename T>
  static Napi::Value GetProperty(const Napi::CallbackInfo& info, Class* cls) {
    T Class::*member = *reinterpret_cast<T Class::**>(info.Data());
    return internal::ToJSValue(info, property_cast(cls->*member));
  }

  template <typename Class, typename T>
  static void SetProperty(const Napi::CallbackInfo& info, Class* cls,
                          const Napi::Value& value) {
    using Tuple = ConvertTypeListToDeclarableTuple<base::internal::TypeList<T>>;
    Tuple v;
    if (!internal::ToNativeValue(value, &std::get<0>(v))) {
      NAPI_THROW(InvalidArgument0(info.Env()));
    }
    T Class::*member = *reinterpret_cast<T Class::**>(info.Data());
    cls->*member = std::move(std::get<0>(v));
  }

 private:
  template <typename Functor, typename Tuple,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::ExtractArgs<RunType>>
  static ReturnType CallHelper(Functor&& f, Tuple&& args) {
    return DoCall(std::forward<Functor>(f), std::forward<Tuple>(args),
                  std::make_index_sequence<
                      std::tuple_size<std::decay_t<Tuple>>::value>{});
  }

  template <typename Functor, typename Tuple, size_t... Is,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::ExtractArgs<RunType>>
  static ReturnType DoCall(Functor&& f, Tuple&& args,
                           std::index_sequence<Is...>) {
    using OriginalTuple = base::internal::ConvertTypeListToTuple<ArgList>;
    return f(arg_cast<std::tuple_element_t<Is, OriginalTuple>>(
        std::move(std::get<Is>(args)))...);
  }

  template <typename Functor, typename Class, typename Tuple,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::DropTypeListItem<
                1, base::internal::ExtractArgs<RunType>>>
  static ReturnType CallHelper(Functor&& f, Class* cls, Tuple&& args) {
    return DoCall(std::forward<Functor>(f), cls, std::forward<Tuple>(args),
                  std::make_index_sequence<
                      std::tuple_size<std::decay_t<Tuple>>::value>{});
  }

  template <typename Functor, typename Class, typename Tuple, size_t... Is,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename RunType = typename FunctorTraits::RunType,
            typename ReturnType = typename FunctorTraits::ReturnType,
            typename ArgList = base::internal::DropTypeListItem<
                1, base::internal::ExtractArgs<RunType>>>
  static ReturnType DoCall(Functor&& f, Class* cls, Tuple&& args,
                           std::index_sequence<Is...>) {
    using OriginalTuple = base::internal::ConvertTypeListToTuple<ArgList>;
    return (cls->*f)(arg_cast<std::tuple_element_t<Is, OriginalTuple>>(
        std::move(std::get<Is>(args)))...);
  }

#undef property_cast
#undef ret_cast
#undef arg_cast
#undef PRE_CALL
};

}  // namespace tachyon::node

#endif  // TACHYON_NODE_BASE_NODE_CPP_CALLABLE_H_
