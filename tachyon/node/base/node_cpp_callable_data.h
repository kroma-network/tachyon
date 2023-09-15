#ifndef TACHYON_NODE_BASE_NODE_CPP_CALLABLE_DATA_H_
#define TACHYON_NODE_BASE_NODE_CPP_CALLABLE_DATA_H_

#if defined(TACHYON_NODE_BINDING)

#include <type_traits>
#include <vector>

#include "tachyon/export.h"

namespace tachyon::node {

struct TACHYON_EXPORT NodeCppCallableDataBase {
  std::vector<void*> default_args;

  void Set() {}

  template <typename Arg, typename AllocatableArg = std::decay_t<Arg>>
  void Set(Arg&& arg) {
    default_args.push_back(new AllocatableArg(std::move(arg)));
  }

  template <typename Arg, typename... DefaultArgs>
  void Set(Arg&& arg, DefaultArgs&&... default_args) {
    Set(std::forward<Arg>(arg));
    Set(std::forward<DefaultArgs>(default_args)...);
  }
};

struct TACHYON_EXPORT NodeCppFunctionData : NodeCppCallableDataBase {
  void* function = nullptr;
};

struct TACHYON_EXPORT NodeCppConstructorData : NodeCppCallableDataBase {
  void* cpp_function = nullptr;
  void* js_function = nullptr;
};

template <typename Method>
struct NodeCppMethodData : NodeCppCallableDataBase {
  ~NodeCppMethodData() { delete method; }

  Method* method = nullptr;
  bool is_const = false;
};

template <typename Getter, typename Setter>
struct NodeCppPropertyAccessorData : public NodeCppCallableDataBase {
  ~NodeCppPropertyAccessorData() {
    delete getter;
    delete setter;
  }

  Getter* getter = nullptr;
  Setter* setter = nullptr;
};

struct TACHYON_EXPORT NodeCppStaticPropertyAccessorData
    : public NodeCppCallableDataBase {
  void* getter = nullptr;
  void* setter = nullptr;
};

}  // namespace tachyon::node

#endif  // defined(TACHYON_NODE_BINDING)

#endif  // TACHYON_NODE_BASE_NODE_CPP_CALLABLE_DATA_H_
