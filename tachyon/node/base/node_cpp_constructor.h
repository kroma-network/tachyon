#ifndef TACHYON_NODE_BASE_NODE_CPP_CONSTRUCTOR_H_
#define TACHYON_NODE_BASE_NODE_CPP_CONSTRUCTOR_H_

#if defined(TACHYON_NODE_BINDING)

#include <string>
#include <vector>

#include "third_party/node_addon_api/napi.h"

#include "tachyon/base/binding/cpp_constructor.h"
#include "tachyon/base/binding/cpp_type_names.h"
#include "tachyon/base/binding/cpp_value_factory.h"
#include "tachyon/base/binding/holder_util.h"
#include "tachyon/node/base/node_internals_forwards.h"

namespace tachyon::node {

namespace internal {

template <typename T>
constexpr bool should_override_to_number_v = std::is_enum<T>::value ||
                                             (std::is_integral<T>::value &&
                                              sizeof(T) <= 4 &&
                                              !std::is_same<T, bool>::value);

}  // namespace internal

class TACHYON_EXPORT NodeCppConstructor : public base::CppConstructor {
 public:
  // TODO(chokobole): Need to use universal reference and perfect forwarding
  template <typename Holder, typename Class, typename... Args>
  static std::unique_ptr<base::CppValue> Create(Args... args) {
    return base::CppValueFactory<Holder>::Create(
        base::internal::HolderCreator<Holder, Class(Args...)>::DoCreate(
            std::forward<Args>(args)...));
  }

  NodeCppConstructor();
  NodeCppConstructor(const NodeCppConstructor& other) = delete;
  NodeCppConstructor& operator=(const NodeCppConstructor& other) = delete;
  NodeCppConstructor(NodeCppConstructor&& other) noexcept;
  NodeCppConstructor& operator=(NodeCppConstructor&& other) noexcept;
  ~NodeCppConstructor();

  template <
      typename Arg,
      std::enable_if_t<internal::should_override_to_number_v<Arg>>* = nullptr>
  void AddArgumentTypename() {
    arg_type_names_.push_back(base::kCppNumberTypeName);
  }

  template <
      typename Arg,
      std::enable_if_t<!internal::should_override_to_number_v<Arg>>* = nullptr>
  void AddArgumentTypename() {
    arg_type_names_.push_back(
        internal::CppValueTraits<internal::NativeCppType<Arg>>::GetTypeName());
  }

  bool Match(const Napi::CallbackInfo& info) const;
};

}  // namespace tachyon::node

#endif  // defined(TACHYON_NODE_BINDING)

#endif  // TACHYON_NODE_BASE_NODE_CPP_CONSTRUCTOR_H_
