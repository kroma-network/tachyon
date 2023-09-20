#include "tachyon/node/base/test/color.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  tachyon::node::NodeModule module(env, exports);
  AddColor(module);
  return exports;
}

NODE_API_MODULE(enum, Init)
