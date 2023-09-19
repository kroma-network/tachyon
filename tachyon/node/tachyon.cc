#if defined(TACHYON_NODE_BINDING)

#include "tachyon/node/math/math.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  tachyon::node::NodeModule m(env, exports);
  tachyon::node::NodeModule math = m.AddSubModule("math");
  tachyon::node::math::AddMath(math);
  return exports;
}

NODE_API_MODULE(tachyon, Init)

#endif  // defined(TACHYON_NODE_BINDING)
