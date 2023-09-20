#ifndef TACHYON_NODE_BASE_NODE_CPP_CONSTRUCTOR_MATCHER_H_
#define TACHYON_NODE_BASE_NODE_CPP_CONSTRUCTOR_MATCHER_H_

#include "third_party/node_addon_api/napi.h"

#include "tachyon/base/binding/cpp_constructor_matcher.h"
#include "tachyon/base/no_destructor.h"

namespace tachyon::node {

template <typename Class>
class NodeCppConstructorMatcher : public base::CppConstructorMatcher {
 public:
  constexpr static size_t kInvalidIndex = std::numeric_limits<size_t>::max();

  static NodeCppConstructorMatcher& GetInstance() {
    static base::NoDestructor<NodeCppConstructorMatcher> matcher;
    return *matcher;
  }

  NodeCppConstructorMatcher(const NodeCppConstructorMatcher& other) = delete;
  NodeCppConstructorMatcher& operator=(const NodeCppConstructorMatcher& other) =
      delete;
  ~NodeCppConstructorMatcher() = default;

  size_t FindConstructorIndex(const Napi::CallbackInfo& info) const {
    for (size_t i = 0; i < constructors_.size(); ++i) {
      NodeCppConstructor* node_cpp_constructor =
          reinterpret_cast<NodeCppConstructor*>(constructors_[i].get());
      if (node_cpp_constructor->Match(info)) {
        return i;
      }
    }
    return kInvalidIndex;
  }

 private:
  friend base::NoDestructor<NodeCppConstructorMatcher>;

  NodeCppConstructorMatcher() = default;
};

}  // namespace tachyon::node

#endif  // TACHYON_NODE_BASE_NODE_CPP_CONSTRUCTOR_MATCHER_H_
