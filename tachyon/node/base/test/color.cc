#if defined(TACHYON_NODE_BINDING)

#include "tachyon/node/base/test/color.h"

#include "tachyon/base/binding/test/color.h"

void AddColor(tachyon::node::NodeModule& m) {
  using namespace tachyon::base::test;
  m.NewEnum<Color>("color")
      .AddValue("red", Color::kRed)
      .AddValue("green", Color::kGreen)
      .AddValue("blue", Color::kBlue);
}

#endif  // defined(TACHYON_NODE_BINDING)
