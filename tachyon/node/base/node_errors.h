#ifndef TACHYON_NODE_BASE_NODE_ERRORS_H_
#define TACHYON_NODE_BASE_NODE_ERRORS_H_

#if defined(TACHYON_NODE_BINDING)

#include <stddef.h>

#include "third_party/node_addon_api/napi.h"

#include "tachyon/export.h"

namespace tachyon::node {

TACHYON_EXPORT Napi::Error WrongNumberOfArguments(Napi::Env env);
TACHYON_EXPORT Napi::Error InvalidArgument(Napi::Env env, size_t n);
TACHYON_EXPORT Napi::Error InvalidArgument0(Napi::Env env);
TACHYON_EXPORT Napi::Error CallNonConstMethod(Napi::Env env);
TACHYON_EXPORT Napi::Error NoSuchConstructor(Napi::Env env);

}  // namespace tachyon::node

#endif  // defined(TACHYON_NODE_BINDING)

#endif  // TACHYON_NODE_BASE_NODE_ERRORS_H_
