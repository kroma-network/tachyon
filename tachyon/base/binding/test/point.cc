#include "tachyon/base/binding/test/point.h"

namespace tachyon::base::test {

Point::Point() = default;
Point::Point(int x, int y) : x(x), y(y) {}

Point::~Point() = default;

// static
double Point::Distance(const Point& p1, const Point& p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

std::string Point::ToString() const {
  return absl::Substitute("($0, $1)", x, y);
}

int Point::s_dimension = 2;

}  // namespace tachyon::base::test
