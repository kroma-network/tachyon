#include "tachyon/node/base/node_module.h"

namespace tachyon::node {

NodeModule::NodeModule(Napi::Env env, Napi::Object exports)
    : NodeCppBindable(env, exports) {}

NodeModule::NodeModule(Napi::Env env, Napi::Object exports,
                       std::string_view name, std::string_view parent_full_name)
    : NodeCppBindable(env, exports, name, parent_full_name) {}

NodeModule::NodeModule(NodeModule&& other) noexcept = default;

NodeModule& NodeModule::operator=(NodeModule&& other) noexcept = default;

NodeModule::~NodeModule() = default;

NodeModule NodeModule::AddSubModule(std::string_view name) {
  Napi::Object exports = Napi::Object::New(env_);
  exports_.Set(name.data(), exports);
  return NodeModule(env_, exports, name, full_name_);
}

}  // namespace tachyon::node
