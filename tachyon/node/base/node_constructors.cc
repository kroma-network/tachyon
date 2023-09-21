#include "tachyon/node/base/node_constructors.h"

namespace tachyon::node {

// static
NodeConstructors& NodeConstructors::GetInstance() {
  static base::NoDestructor<NodeConstructors> node_constructors;
  return *node_constructors;
}

NodeConstructors::NodeConstructors() = default;

NodeConstructors::~NodeConstructors() = default;

void NodeConstructors::AddConstructor(std::string_view type,
                                      Napi::FunctionReference constructor) {
  constructors_[type] = std::move(constructor);
}

bool NodeConstructors::GetConstructor(std::string_view type,
                                      Napi::Function* constructor) const {
  auto it = constructors_.find(type);
  if (it == constructors_.end()) return false;
  if (it->second.IsEmpty()) return false;
  *constructor = it->second.Value();
  return true;
}

bool NodeConstructors::InstanceOf(Napi::Object object, std::string_view name) {
  Napi::Function constructor;
  if (!GetConstructor(name, &constructor)) return false;
  return object.InstanceOf(constructor);
}

}  // namespace tachyon::node
