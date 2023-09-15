#ifndef TACHYON_NODE_BASE_NODE_CONSTRUCTORS_H_
#define TACHYON_NODE_BASE_NODE_CONSTRUCTORS_H_

#if defined(TACHYON_NODE_BINDING)

#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "third_party/node_addon_api/napi.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/export.h"

namespace tachyon::node {

class TACHYON_EXPORT NodeConstructors {
 public:
  static NodeConstructors& GetInstance();

  NodeConstructors(const NodeConstructors& other) = delete;
  NodeConstructors& operator=(const NodeConstructors& other) = delete;
  ~NodeConstructors();

  void AddConstructor(std::string_view type,
                      Napi::FunctionReference constructor);
  bool GetConstructor(std::string_view type, Napi::Function* constructor) const;

  bool InstanceOf(Napi::Object object, std::string_view name);

 private:
  friend class base::NoDestructor<NodeConstructors>;

  NodeConstructors();

  absl::flat_hash_map<std::string, Napi::FunctionReference> constructors_;
};

}  // namespace tachyon::node

#endif  // defined(TACHYON_NODE_BINDING)

#endif  // TACHYON_NODE_BASE_NODE_CONSTRUCTORS_H_
