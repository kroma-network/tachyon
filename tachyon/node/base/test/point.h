#if defined(TACHYON_NODE_BINDING)

#include <string_view>

#include "tachyon/base/binding/test/point.h"
#include "tachyon/node/base/node_module.h"

template <typename Holder>
void AddPointImpl(tachyon::node::NodeModule& m, std::string_view name) {
  using namespace tachyon::base::test;
  m.NewClass<Point, Holder>(name)
      .template AddConstructor<>()
      .template AddConstructor<int, int>()
      .AddStaticMethod("getDimension", &Point::GetDimension)
      .AddStaticMethod("setDimension", &Point::SetDimension)
      .AddStaticMethod("distance", &Point::Distance)
      .AddStaticProperty("propertyDimension", &Point::GetDimension,
                         &Point::SetDimension)
      .AddStaticReadWrite("dimension", &Point::s_dimension)
      .AddMethod("getX", &Point::GetX)
      .AddMethod("setX", &Point::SetX)
      .AddMethod("getY", &Point::GetY)
      .AddMethod("setY", &Point::SetY)
      .AddProperty("propertyX", &Point::GetX, &Point::SetX)
      .AddProperty("propertyY", &Point::GetY, &Point::SetY)
      .AddReadWrite("x", &Point::x)
      .AddReadWrite("y", &Point::y);
}

void AddPoint(tachyon::node::NodeModule& m);

void AddSharedPoint(tachyon::node::NodeModule& m);

void AddUniquePoint(tachyon::node::NodeModule& m);

tachyon::base::test::Point DoubleWithValue(tachyon::base::test::Point p);

void DoubleWithReference(tachyon::base::test::Point& p);

void DoubleWithSharedPtr(std::shared_ptr<tachyon::base::test::Point> p);

void DoubleWithUniquePtr(std::unique_ptr<tachyon::base::test::Point> p);

#endif  // defined(TACHYON_NODE_BINDING)
