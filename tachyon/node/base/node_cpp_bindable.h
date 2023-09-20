#ifndef TACHYON_NODE_BASE_NODE_CPP_BINDABLE_H_
#define TACHYON_NODE_BASE_NODE_CPP_BINDABLE_H_

#include <string>
#include <string_view>

#include "third_party/node_addon_api/napi.h"

#include "tachyon/export.h"

namespace tachyon::node {

class TACHYON_EXPORT NodeCppBindable {
 public:
  NodeCppBindable(Napi::Env env, Napi::Object exports);
  NodeCppBindable(Napi::Env env, Napi::Object exports, std::string_view name,
                  std::string_view parent_full_name);
  NodeCppBindable(const NodeCppBindable& other) = delete;
  NodeCppBindable& operator=(const NodeCppBindable& other) = delete;
  NodeCppBindable(NodeCppBindable&& other) noexcept;
  NodeCppBindable& operator=(NodeCppBindable&& other) noexcept;
  ~NodeCppBindable();

 protected:
  Napi::Env env_;
  Napi::Object exports_;
  std::string_view name_;
  std::string full_name_;
};

}  // namespace tachyon::node

#endif  // TACHYON_NODE_BASE_NODE_CPP_BINDABLE_H_
