#ifndef TACHYON_NODE_BASE_NODE_MODULE_H_
#define TACHYON_NODE_BASE_NODE_MODULE_H_

#include <utility>

#include "tachyon/export.h"
#include "tachyon/node/base/node_cpp_bindable.h"
#include "tachyon/node/base/node_cpp_callable.h"
#include "tachyon/node/base/node_cpp_callable_data.h"
#include "tachyon/node/base/node_cpp_class.h"
#include "tachyon/node/base/node_cpp_enum.h"
#include "tachyon/node/base/node_internals.h"

namespace tachyon::node {

class TACHYON_EXPORT NodeModule : public NodeCppBindable {
 public:
  NodeModule(Napi::Env env, Napi::Object exports);
  NodeModule(const NodeModule& other) = delete;
  NodeModule& operator=(const NodeModule& other) = delete;
  NodeModule(NodeModule&& other) noexcept;
  NodeModule& operator=(NodeModule&& other) noexcept;
  ~NodeModule();

  NodeModule AddSubModule(std::string_view name);

  template <typename R, typename... Args, typename... DefaultArgs>
  NodeModule& AddFunction(std::string_view name, R (*f)(Args...),
                          DefaultArgs&&... default_args) {
    static_assert(sizeof...(Args) >= sizeof...(DefaultArgs),
                  "Too many default args");
    using Function = R (*)(Args...);
    NodeCppFunctionData* data = new NodeCppFunctionData;
    data->function = reinterpret_cast<void*>(f);
    data->Set(std::forward<DefaultArgs>(default_args)...);
    exports_.Set(name.data(),
                 Napi::Function::New<NodeCppCallable::Call<Function>>(
                     env_, name.data(), data));
    return *this;
  }

  template <typename Class, typename... Options>
  NodeCppClass<Class, Options...> NewClass(std::string_view name) {
    return NodeCppClass<Class, Options...>(env_, exports_, name, full_name_);
  }

  template <typename Enum>
  NodeCppEnum<Enum> NewEnum(std::string_view name) {
    return NodeCppEnum<Enum>(env_, exports_, name, full_name_);
  }

 private:
  NodeModule(Napi::Env env, Napi::Object exports, std::string_view name,
             std::string_view parent_full_name);
};

}  // namespace tachyon::node

#endif  // TACHYON_NODE_BASE_NODE_MODULE_H_
