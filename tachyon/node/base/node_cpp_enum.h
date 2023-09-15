#ifndef TACHYON_NODE_BASE_NODE_CPP_ENUM_H_
#define TACHYON_NODE_BASE_NODE_CPP_ENUM_H_

#if defined(TACHYON_NODE_BINDING)

#include <string_view>
#include <type_traits>

#include "tachyon/base/binding/cpp_type.h"
#include "tachyon/node/base/node_cpp_bindable.h"

namespace tachyon::node {

template <typename Enum>
class NodeCppEnum : public NodeCppBindable {
 public:
  NodeCppEnum(const NodeCppEnum& other) = delete;
  NodeCppEnum& operator=(const NodeCppEnum& other) = delete;
  NodeCppEnum(NodeCppEnum&& other) noexcept = default;
  NodeCppEnum& operator=(NodeCppEnum&& other) noexcept = default;
  ~NodeCppEnum() {
    // TODO(chokobole): Need to freeze |env_|.
  }

  NodeCppEnum& AddValue(std::string_view name, Enum value) {
    enum_.DefineProperty(Napi::PropertyDescriptor::Value(
        name.data(),
        Napi::Number::New(env_,
                          static_cast<std::underlying_type_t<Enum>>(value)),
        napi_enumerable));
    return *this;
  }

 private:
  friend class NodeModule;

  NodeCppEnum(Napi::Env env, Napi::Object exports, std::string_view name,
              std::string_view parent_full_name)
      : NodeCppBindable(env, exports, name, parent_full_name) {
    base::CppType<Enum>::Get().set_name(full_name_);
    enum_ = Napi::Object::New(env_);
    exports_.Set(name.data(), enum_);
  }

  Napi::Object enum_;
};

}  // namespace tachyon::node

#endif  // defined(TACHYON_NODE_BINDING)

#endif  // TACHYON_NODE_BASE_NODE_CPP_ENUM_H_
