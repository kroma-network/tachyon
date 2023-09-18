#if defined(TACHYON_NODE_BINDING)

#include "tachyon/base/binding/test/functions.h"
#include "tachyon/node/base/node_module.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  using namespace tachyon::base::test;
  tachyon::node::NodeModule module(env, exports);
  module.AddFunction("hello", &Hello)
      .AddFunction("sum", &Sum, 1, 2)
      .AddFunction("do_nothing", &DoNothing);
  return exports;
}

NODE_API_MODULE(function, Init)

#endif  // defined(TACHYON_NODE_BINDING)
