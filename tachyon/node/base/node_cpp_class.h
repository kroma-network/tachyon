#ifndef TACHYON_NODE_BASE_NODE_CPP_CLASS_H_
#define TACHYON_NODE_BASE_NODE_CPP_CLASS_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/binding/cpp_type.h"
#include "tachyon/base/binding/holder_util.h"
#include "tachyon/node/base/node_cpp_bindable.h"
#include "tachyon/node/base/node_cpp_callable_data.h"
#include "tachyon/node/base/node_cpp_constructor.h"
#include "tachyon/node/base/node_cpp_constructor_matcher.h"
#include "tachyon/node/base/node_cpp_enum.h"
#include "tachyon/node/base/node_cpp_object.h"

namespace tachyon::node {

template <typename Class, typename... Options>
class NodeCppClass : public NodeCppBindable {
  template <typename T>
  using is_holder = base::internal::is_holder_type<Class, T>;
  template <typename T>
  using is_base = base::is_strict_base_of<T, Class>;

  template <typename T>
  struct is_valid_class_option : base::any_of<is_holder<T>, is_base<T>> {};

 public:
  static_assert(base::all_of<is_valid_class_option<Options>...>::value,
                "Unknown/invalid NodeCppClass template parameters provided");

  using Holder = base::exactly_one_t<is_holder, Class, Options...>;

  using Object = NodeCppObject<Class>;

  NodeCppClass(const NodeCppClass& other) = delete;
  NodeCppClass& operator=(const NodeCppClass& other) = delete;
  NodeCppClass(NodeCppClass&& other) noexcept = default;
  NodeCppClass& operator=(NodeCppClass&& other) noexcept = default;
  ~NodeCppClass() {
    Object::Init(env_, exports_, name_, full_name_, properties_,
                 constructor_data_vec_);
    TACHYON_EXPAND_SIDE_EFFECTS(RegisterParent<Options>());
  }

  template <typename... Args, typename... DefaultArgs>
  NodeCppClass& AddConstructor(DefaultArgs&&... default_args) {
    static_assert(sizeof...(Args) >= sizeof...(DefaultArgs),
                  "Too many default args");
    if (!constructor_data_vec_) {
      constructor_data_vec_ = new std::vector<NodeCppConstructorData>();
    }

    NodeCppConstructorMatcher<Class>& constructor_matcher =
        NodeCppConstructorMatcher<Class>::GetInstance();
    std::unique_ptr<NodeCppConstructor> constructor(new NodeCppConstructor());
    TACHYON_EXPAND_SIDE_EFFECTS(constructor->AddArgumentTypename<Args>());
    constructor->ValidateAndMaybeDie();
    constructor->SetDefaultArgsNum(sizeof...(DefaultArgs));
    constructor_matcher.Add(std::move(constructor));

    NodeCppConstructorData data;
    data.cpp_function = reinterpret_cast<void*>(
        &NodeCppConstructor::Create<Holder, Class, Args...>);
    data.js_function = reinterpret_cast<void*>(
        &NodeCppCallable::CallConstructor<std::unique_ptr<base::CppValue> (*)(
            Args...)>);
    data.Set(std::forward<DefaultArgs>(default_args)...);
    constructor_data_vec_->push_back(std::move(data));
    return *this;
  }

  template <typename T>
  NodeCppClass& AddStaticReadOnly(std::string_view name, T* value) {
    return AddStaticReadOnlyWithAttributes(name, value, napi_default);
  }

  template <typename T>
  NodeCppClass& AddStaticReadOnlyWithAttributes(
      std::string_view name, T* value, napi_property_attributes attributes) {
    properties_.push_back(Napi::ObjectWrap<Object>::StaticAccessor(
        name.data(), &NodeCppCallable::GetStaticProperty<T>, nullptr,
        attributes, value));
    return *this;
  }

  template <typename T>
  NodeCppClass& AddStaticReadWrite(std::string_view name, T* value) {
    return AddStaticReadWriteWithAttributes(name, value, napi_default);
  }

  template <typename T>
  NodeCppClass& AddStaticReadWriteWithAttributes(
      std::string_view name, T* value, napi_property_attributes attributes) {
    properties_.push_back(Napi::ObjectWrap<Object>::StaticAccessor(
        name.data(), &NodeCppCallable::GetStaticProperty<T>,
        &NodeCppCallable::SetStaticProperty<T>, attributes, value));
    return *this;
  }

  template <typename T>
  NodeCppClass& AddStaticReadOnlyProperty(std::string_view name,
                                          T (*getter)()) {
    return AddStaticReadOnlyPropertyWithAttributes(name, getter, napi_default);
  }

  template <typename T>
  NodeCppClass& AddStaticReadOnlyPropertyWithAttributes(
      std::string_view name, T (*getter)(),
      napi_property_attributes attributes) {
    return AddStaticPropertyWithAttributes(name, getter, nullptr, attributes);
  }

  template <typename T, typename U>
  NodeCppClass& AddStaticProperty(std::string_view name, T (*getter)(),
                                  void (*setter)(U)) {
    return AddStaticPropertyWithAttributes(name, getter, setter, napi_default);
  }

  template <typename T, typename U>
  NodeCppClass& AddStaticPropertyWithAttributes(
      std::string_view name, T (*getter)(), void (*setter)(U),
      napi_property_attributes attributes) {
    using Getter = T (*)();
    using Setter = void (*)(U);

    NodeCppStaticPropertyAccessorData* accessor =
        new NodeCppStaticPropertyAccessorData;
    accessor->getter = reinterpret_cast<void*>(getter);
    accessor->setter = reinterpret_cast<void*>(setter);
    properties_.push_back(Napi::ObjectWrap<Object>::StaticAccessor(
        name.data(), &NodeCppCallable::CallStaticGetter<Getter>,
        setter ? &NodeCppCallable::CallStaticSetter<Setter> : nullptr,
        attributes, accessor));
    return *this;
  }

  template <typename R, typename... Args, typename... DefaultArgs>
  NodeCppClass& AddStaticMethod(std::string_view name, R (*f)(Args...),
                                DefaultArgs&&... default_args) {
    return AddStaticMethodWithAttributes(
        name, f, napi_default, std::forward<DefaultArgs>(default_args)...);
  }

  template <typename R, typename... Args, typename... DefaultArgs>
  NodeCppClass& AddStaticMethodWithAttributes(
      std::string_view name, R (*f)(Args...),
      napi_property_attributes attributes, DefaultArgs&&... default_args) {
    static_assert(sizeof...(Args) >= sizeof...(DefaultArgs),
                  "Too many default args");
    using StaticMethod = R (*)(Args...);

    NodeCppFunctionData* data = new NodeCppFunctionData;
    data->function = reinterpret_cast<void*>(f);
    data->Set(std::forward<DefaultArgs>(default_args)...);
    properties_.push_back(Napi::ObjectWrap<Object>::StaticMethod(
        name.data(), &NodeCppCallable::template Call<StaticMethod>, attributes,
        data));
    return *this;
  }

  template <
      typename T, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddReadOnly(std::string_view name, T BaseClass::*value) {
    return AddReadOnlyWithAttributes(name, value, napi_default);
  }

  template <
      typename T, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddReadOnlyWithAttributes(std::string_view name,
                                          T BaseClass::*value,
                                          napi_property_attributes attributes) {
    return DoAddProperty(name, value, attributes, /*needs_setter=*/false);
  }

  template <
      typename T, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddReadWrite(std::string_view name, T BaseClass::*value) {
    return AddReadWriteWithAttributes(name, value, napi_default);
  }

  template <
      typename T, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddReadWriteWithAttributes(
      std::string_view name, T BaseClass::*value,
      napi_property_attributes attributes) {
    return DoAddProperty(name, value, attributes, /*needs_setter=*/true);
  }

  template <
      typename T, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddReadOnlyProperty(std::string_view name,
                                    T (BaseClass::*getter)() const) {
    return AddReadOnlyPropertyWithAttributes(name, getter, napi_default);
  }

  template <
      typename T, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddReadOnlyPropertyWithAttributes(
      std::string_view name, T (BaseClass::*getter)() const,
      napi_property_attributes attributes) {
    using Getter = T (BaseClass::*)() const;
    using PropertyAccessorData = NodeCppPropertyAccessorData<Getter, void>;

    PropertyAccessorData* accessor = new PropertyAccessorData;
    accessor->getter = new Getter(getter);
    accessor->setter = nullptr;
    properties_.push_back(Napi::InstanceWrap<Object>::InstanceAccessor(
        name.data(), &Object::template CallGetter<Getter, void>, nullptr,
        attributes, accessor));
    return *this;
  }

  template <
      typename T, typename U, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddProperty(std::string_view name,
                            T (BaseClass::*getter)() const,
                            void (BaseClass::*setter)(U)) {
    return AddPropertyWithAttributes(name, getter, setter, napi_default);
  }

  template <
      typename T, typename U, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddPropertyWithAttributes(std::string_view name,
                                          T (BaseClass::*getter)() const,
                                          void (BaseClass::*setter)(U),
                                          napi_property_attributes attributes) {
    using Getter = T (BaseClass::*)() const;
    using Setter = void (BaseClass::*)(U);
    using PropertyAccessorData = NodeCppPropertyAccessorData<Getter, Setter>;

    PropertyAccessorData* accessor = new PropertyAccessorData;
    accessor->getter = new Getter(getter);
    accessor->setter = new Setter(setter);
    properties_.push_back(Napi::InstanceWrap<Object>::InstanceAccessor(
        name.data(), &Object::template CallGetter<Getter, Setter>,
        &Object::template CallSetter<Getter, Setter>, attributes, accessor));
    return *this;
  }

  template <
      typename R, typename BaseClass, typename... Args, typename... DefaultArgs,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddMethod(std::string_view name,
                          R (BaseClass::*method)(Args...),
                          DefaultArgs&&... default_args) {
    return AddMethodWithAttributes(name, method, napi_default,
                                   std::forward<DefaultArgs>(default_args)...);
  }

  template <
      typename R, typename BaseClass, typename... Args, typename... DefaultArgs,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddMethodWithAttributes(std::string_view name,
                                        R (BaseClass::*method)(Args...),
                                        napi_property_attributes attributes,
                                        DefaultArgs&&... default_args) {
    static_assert(sizeof...(Args) >= sizeof...(DefaultArgs),
                  "Too many default args");
    return DoAddMethod(name, method, attributes, false,
                       std::forward<DefaultArgs>(default_args)...);
  }

  template <
      typename R, typename BaseClass, typename... Args, typename... DefaultArgs,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddMethod(std::string_view name,
                          R (BaseClass::*method)(Args...) const,
                          DefaultArgs&&... default_args) {
    return AddMethodWithAttributes(name, method, napi_default,
                                   std::forward<DefaultArgs>(default_args)...);
  }

  template <
      typename R, typename BaseClass, typename... Args, typename... DefaultArgs,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& AddMethodWithAttributes(std::string_view name,
                                        R (BaseClass::*method)(Args...) const,
                                        napi_property_attributes attributes,
                                        DefaultArgs&&... default_args) {
    static_assert(sizeof...(Args) >= sizeof...(DefaultArgs),
                  "Too many default args");
    return DoAddMethod(name, method, attributes, true,
                       std::forward<DefaultArgs>(default_args)...);
  }

  template <typename SubClass, typename... SubClassOptions>
  NodeCppClass<SubClass, SubClassOptions...> NewClass(std::string_view name) {
    return NodeCppClass<SubClass, SubClassOptions...>(env_, exports_, name,
                                                      full_name_);
  }

  template <typename Enum>
  NodeCppEnum<Enum> NewEnum(std::string_view name) {
    return NodeCppEnum<Enum>(env_, exports_, name, full_name_);
  }

 private:
  friend class NodeModule;

  NodeCppClass(Napi::Env env, Napi::Object exports, std::string_view name,
               std::string_view parent_full_name)
      : NodeCppBindable(env, exports, name, parent_full_name) {
    base::CppType<Class>::Get().set_name(full_name_);
    base::CppType<const Class>::Get().set_name(
        absl::Substitute("const $0", full_name_));
  }

  template <typename BaseClass,
            std::enable_if_t<!is_base<BaseClass>::value>* = nullptr>
  void RegisterParent() {}

  template <typename BaseClass,
            std::enable_if_t<is_base<BaseClass>::value>* = nullptr>
  void RegisterParent() {
    NodeConstructors& constructors = NodeConstructors::GetInstance();
    const std::string& parent_full_name =
        base::CppType<BaseClass>::Get().name();
    Napi::Function parent_constructor;
    CHECK(constructors.GetConstructor(parent_full_name, &parent_constructor))
        << "Did you forget to register parent: " << parent_full_name << "?";
    Object::RegisterParent(parent_constructor);
  }

  template <
      typename T, typename BaseClass,
      std::enable_if_t<std::is_base_of<BaseClass, Class>::value>* = nullptr>
  NodeCppClass& DoAddProperty(std::string_view name, T BaseClass::*value,
                              napi_property_attributes attributes,
                              bool needs_setter) {
    using Member = T BaseClass::*;

    Member* member = new Member(value);
    properties_.push_back(Napi::InstanceWrap<Object>::InstanceAccessor(
        name.data(), &Object::template GetProperty<T>,
        needs_setter ? &Object::template SetProperty<T> : nullptr, attributes,
        member));
    return *this;
  }

  template <typename Method, typename... DefaultArgs>
  NodeCppClass& DoAddMethod(std::string_view name, Method method,
                            napi_property_attributes attributes, bool is_const,
                            DefaultArgs&&... default_args) {
    NodeCppMethodData<Method>* data = new NodeCppMethodData<Method>;
    data->method = new Method(method);
    data->is_const = is_const;
    data->Set(std::forward<DefaultArgs>(default_args)...);
    properties_.push_back(Napi::InstanceWrap<Object>::InstanceMethod(
        name.data(), &Object::template CallMethod<Method>, attributes, data));
    return *this;
  }

  std::vector<NodeCppConstructorData>* constructor_data_vec_ = nullptr;
  std::vector<Napi::ClassPropertyDescriptor<Object>> properties_;
};

}  // namespace tachyon::node

#endif  // TACHYON_NODE_BASE_NODE_CPP_CLASS_H_
