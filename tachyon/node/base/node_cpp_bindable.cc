#if defined(TACHYON_NODE_BINDING)

#include "tachyon/node/base/node_cpp_bindable.h"

#include "absl/strings/substitute.h"

namespace tachyon::node {

NodeCppBindable::NodeCppBindable(Napi::Env env, Napi::Object exports)
    : env_(env), exports_(exports) {}

NodeCppBindable::NodeCppBindable(Napi::Env env, Napi::Object exports,
                                 std::string_view name,
                                 std::string_view parent_full_name)
    : env_(env),
      exports_(exports),
      name_(name),
      full_name_(parent_full_name.empty()
                     ? std::string(name)
                     : absl::Substitute("$0.$1", parent_full_name, name_)) {}

NodeCppBindable::NodeCppBindable(NodeCppBindable&& other) noexcept = default;

NodeCppBindable& NodeCppBindable::operator=(NodeCppBindable&& other) noexcept =
    default;

NodeCppBindable::~NodeCppBindable() = default;

}  // namespace tachyon::node

#endif  // defined(TACHYON_NODE_BINDING)
