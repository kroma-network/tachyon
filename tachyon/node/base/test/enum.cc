#if defined(TACHYON_NODE_BINDING)

#include "tachyon/node/base/test/color.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  tachyon::node::NodeModule module(env, exports);
  AddColor(module);
  return exports;
}

NODE_API_MODULE(enum, Init)

#endif  // defined(TACHYON_NODE_BINDING)
