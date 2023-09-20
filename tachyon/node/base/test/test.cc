#if defined(TACHYON_NODE_BINDING)

#include "tachyon/base/binding/test/adder.h"
#include "tachyon/base/binding/test/colored_point.h"
#include "tachyon/base/binding/test/functions.h"
#include "tachyon/base/binding/test/rect.h"
#include "tachyon/base/binding/test/variant.h"
#include "tachyon/node/base/node_module.h"
#include "tachyon/node/base/test/color.h"
#include "tachyon/node/base/test/point.h"

void AcceptColoredPoint(tachyon::base::test::ColoredPoint& cp) {}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  using namespace tachyon::base::test;

  tachyon::node::NodeModule m(env, exports);

  m.AddFunction("hello", &Hello)
      .AddFunction("sum", &Sum, 1, 2)
      .AddFunction("do_nothing", &DoNothing);

  m.NewEnum<Color>("color")
      .AddValue("red", Color::kRed)
      .AddValue("green", Color::kGreen)
      .AddValue("blue", Color::kBlue);

  // TODO(chokobole): I tried testing all these variation either in same addon
  // or different addon. But it somehow corrupts this addon. So I need to figure
  // out how to gracefully test! Uncomment either one of AddXXXPoint.
  AddPoint(m);
  // This enable 'shared_ptr' test in 'class.spec.ts'.
  // AddSharedPoint(m);
  // This enable 'unique_ptr' test in 'class.spec.ts'.
  // AddUniquePoint(m);

  m.NewClass<ColoredPoint, Point>("ColoredPoint")
      .AddConstructor<>()
      .AddConstructor<int, int, Color>();

  m.NewClass<Rect>("Rect")
      .AddConstructor<>()
      .AddConstructor<const Point&, const Point&>()
      .AddReadWrite("topLeft", &Rect::top_left)
      .AddReadWrite("bottomRight", &Rect::bottom_right)
      .AddMethod("getTopLeft", &Rect::GetTopLeft)
      .AddMethod("getConstTopLeft", &Rect::GetConstTopLeft)
      .AddMethod("getBottomRight", &Rect::GetBottomRight)
      .AddMethod("getConstBottomRight", &Rect::GetConstBottomRight);

  m.NewClass<Adder>("Adder")
      .AddConstructor<>()
      .AddMethod("add", &Adder::Add, 1, 2, 3, 4)
      .AddStaticMethod("sAdd", &Adder::SAdd, 1, 2, 3, 4);

  m.NewClass<Variant>("Variant")
      .AddConstructor<bool>()
      .AddConstructor<int>()
      .AddConstructor<int64_t>()
      .AddConstructor<const std::string&>()
      .AddConstructor<const std::vector<int>&>()
      .AddConstructor<int, const std::string&>()
      .AddConstructor<const Point&>()
      .AddReadWrite("b", &Variant::b)
      .AddReadWrite("i", &Variant::i)
      .AddReadWrite("i64", &Variant::i64)
      .AddReadWrite("s", &Variant::s)
      .AddReadWrite("ivec", &Variant::ivec)
      .AddReadWrite("p", &Variant::p);

  m.AddFunction("doubleWithValue", &DoubleWithValue)
      .AddFunction("doubleWithReference", &DoubleWithReference)
      .AddFunction("doubleWithSharedPtr", &DoubleWithSharedPtr)
      .AddFunction("doubleWithUniquePtr", &DoubleWithUniquePtr)
      .AddFunction("acceptColoredPoint", &AcceptColoredPoint);
  return exports;
}

NODE_API_MODULE(test, Init)

#endif  // defined(TACHYON_NODE_BINDING)
