#ifndef TACHYON_NODE_BASE_NODE_CPP_OBJECT_H_
#define TACHYON_NODE_BASE_NODE_CPP_OBJECT_H_

#include <memory>
#include <string_view>
#include <vector>

#include "tachyon/base/binding/cpp_stack_value.h"
#include "tachyon/base/binding/cpp_value.h"
#include "tachyon/node/base/node_constructors.h"
#include "tachyon/node/base/node_cpp_callable.h"
#include "tachyon/node/base/node_errors.h"

namespace tachyon::node {

template <typename Class>
class NodeCppObject : public Napi::ObjectWrap<NodeCppObject<Class>> {
 public:
  explicit NodeCppObject(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<NodeCppObject>(info) {
    if (info.Length() == 1 && info[0].IsExternal()) {
      value_.reset(info[0].As<Napi::External<base::CppValue>>().Data());
    } else {
      NodeCppConstructorMatcher<Class>& constructor_matcher =
          NodeCppConstructorMatcher<Class>::GetInstance();
      size_t idx = constructor_matcher.FindConstructorIndex(info);
      if (idx == NodeCppConstructorMatcher<Class>::kInvalidIndex) {
        NAPI_THROW(NoSuchConstructor(info.Env()));
      }
      std::vector<NodeCppConstructorData>* data_vec =
          reinterpret_cast<std::vector<NodeCppConstructorData>*>(info.Data());
      NodeCppConstructorData& data = (*data_vec)[idx];
      using JSFunction = std::unique_ptr<base::CppValue> (*)(
          const Napi::CallbackInfo&, NodeCppConstructorData*);
      JSFunction js_function = reinterpret_cast<JSFunction>(data.js_function);
      value_ = js_function(info, &data);
    }
  }
  NodeCppObject(const NodeCppObject& other) = delete;
  NodeCppObject& operator=(const NodeCppObject& other) = delete;
  ~NodeCppObject() = default;

  base::CppValue* value() { return value_.get(); }

  static void Init(
      Napi::Env env, Napi::Object exports, std::string_view name,
      std::string_view full_name,
      std::vector<Napi::ClassPropertyDescriptor<NodeCppObject>>& properties,
      void* data = nullptr) {
    properties.push_back(NodeCppObject::InstanceAccessor(
        "__is_const__", &NodeCppObject::GetIsConst, nullptr));
    Napi::Function func =
        NodeCppObject::DefineClass(env, name.data(), properties, data);
    s_constructor_ = Napi::Persistent(func);
    s_constructor_.SuppressDestruct();
    exports.Set(name.data(), func);
    NodeConstructors& constructors = NodeConstructors::GetInstance();
    constructors.AddConstructor(full_name, Napi::Weak(func));
    constructors.AddConstructor(absl::Substitute("const $0", full_name),
                                Napi::Weak(func));
  }

  static void RegisterParent(Napi::Function parent_constructor) {
    s_constructor_.Value().Set("prototype",
                               parent_constructor.Get("prototype"));
  }

  static Napi::Object NewInstance(Napi::Env env,
                                  Napi::External<base::CppValue> arg) {
    Napi::EscapableHandleScope scope(env);
    Napi::Object obj = s_constructor_.New({arg});
    return scope.Escape(napi_value(obj)).ToObject();
  }

  template <typename Functor,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename ReturnType = typename FunctorTraits::ReturnType,
            std::enable_if_t<std::is_same<ReturnType, void>::value>* = nullptr>
  void CallMethod(const Napi::CallbackInfo& info) {
    NodeCppMethodData<Functor>* data =
        reinterpret_cast<NodeCppMethodData<Functor>*>(info.Data());
    if (value_->is_const() && !data->is_const) {
      NAPI_THROW(CallNonConstMethod(info.Env()));
    }
    NodeCppCallable::CallMethod<Functor>(
        info, reinterpret_cast<Class*>(value_->raw_ptr()));
  }

  template <typename Functor,
            typename FunctorTraits = base::internal::MakeFunctorTraits<Functor>,
            typename ReturnType = typename FunctorTraits::ReturnType,
            std::enable_if_t<!std::is_same<ReturnType, void>::value>* = nullptr>
  Napi::Value CallMethod(const Napi::CallbackInfo& info) {
    NodeCppMethodData<Functor>* data =
        reinterpret_cast<NodeCppMethodData<Functor>*>(info.Data());
    if (value_->is_const() && !data->is_const) {
      NAPI_THROW(CallNonConstMethod(info.Env()), info.Env().Null());
    }
    return NodeCppCallable::CallMethod<Functor>(
        info, reinterpret_cast<Class*>(value_->raw_ptr()));
  }

  template <typename T>
  Napi::Value GetProperty(const Napi::CallbackInfo& info) {
    if (value_->is_const()) {
      return NodeCppCallable::GetProperty<const Class, T>(
          info, reinterpret_cast<const Class*>(value_->raw_ptr()));
    } else {
      return NodeCppCallable::GetProperty<Class, T>(
          info, reinterpret_cast<Class*>(value_->raw_ptr()));
    }
  }

  template <typename T>
  void SetProperty(const Napi::CallbackInfo& info, const Napi::Value& value) {
    if (value_->is_const()) {
      NAPI_THROW(CallNonConstMethod(info.Env()));
    }
    NodeCppCallable::SetProperty<Class, T>(
        info, reinterpret_cast<Class*>(value_->raw_ptr()), value);
  }

  template <typename Getter, typename Setter>
  Napi::Value CallGetter(const Napi::CallbackInfo& info) {
    if (value_->is_const()) {
      return NodeCppCallable::CallGetter<Getter, Setter>(
          info, reinterpret_cast<const Class*>(value_->raw_ptr()));
    } else {
      return NodeCppCallable::CallGetter<Getter, Setter>(
          info, reinterpret_cast<Class*>(value_->raw_ptr()));
    }
  }

  template <typename Getter, typename Setter>
  void CallSetter(const Napi::CallbackInfo& info, const Napi::Value& value) {
    if (value_->is_const()) {
      NAPI_THROW(CallNonConstMethod(info.Env()));
    }
    NodeCppCallable::CallSetter<Getter, Setter>(
        info, reinterpret_cast<Class*>(value_->raw_ptr()), value);
  }

 private:
  Napi::Value GetIsConst(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), value_->is_const());
  }

  static Napi::FunctionReference s_constructor_;
  std::unique_ptr<base::CppValue> value_;
};

template <typename T>
Napi::FunctionReference NodeCppObject<T>::s_constructor_;

}  // namespace tachyon::node

#endif  // TACHYON_NODE_BASE_NODE_CPP_OBJECT_H_
