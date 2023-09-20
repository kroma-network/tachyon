#include "tachyon/node/base/node_errors.h"

#include "absl/strings/substitute.h"

namespace tachyon::node {

Napi::Error WrongNumberOfArguments(Napi::Env env) {
  return Napi::TypeError::New(env, "Wrong number of arguments");
}

Napi::Error InvalidArgument(Napi::Env env, size_t n) {
  return Napi::TypeError::New(env, absl::Substitute("Invalid argument #$0", n));
}

Napi::Error InvalidArgument0(Napi::Env env) {
  return Napi::TypeError::New(env, "Invalid argument #0");
}

Napi::Error CallNonConstMethod(Napi::Env env) {
  return Napi::TypeError::New(env, "Call non-const method");
}

Napi::Error NoSuchConstructor(Napi::Env env) {
  return Napi::TypeError::New(env, "No such constructor");
}

}  // namespace tachyon::node
