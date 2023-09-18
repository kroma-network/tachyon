#if defined(TACHYON_NODE_BINDING)

#include "tachyon/node/base/test/point.h"

void AddPoint(tachyon::node::NodeModule& m) {
  AddPointImpl<tachyon::base::test::Point>(m, "Point");
}

void AddSharedPoint(tachyon::node::NodeModule& m) {
  AddPointImpl<std::shared_ptr<tachyon::base::test::Point>>(m, "Point");
}

void AddUniquePoint(tachyon::node::NodeModule& m) {
  AddPointImpl<std::unique_ptr<tachyon::base::test::Point>>(m, "Point");
}

tachyon::base::test::Point DoubleWithValue(tachyon::base::test::Point p) {
  return {p.x * 2, p.y * 2};
}

void DoubleWithReference(tachyon::base::test::Point& p) {
  p.x *= 2;
  p.y *= 2;
}

void DoubleWithSharedPtr(std::shared_ptr<tachyon::base::test::Point> p) {
  p->x *= 2;
  p->y *= 2;
}

void DoubleWithUniquePtr(std::unique_ptr<tachyon::base::test::Point> p) {}

#endif  // defined(TACHYON_NODE_BINDING)
