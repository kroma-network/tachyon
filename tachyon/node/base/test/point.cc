#include "tachyon/node/base/test/point.h"

#include <memory>

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
