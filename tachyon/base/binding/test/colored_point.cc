#include "tachyon/base/binding/test/colored_point.h"

namespace tachyon::base::test {

ColoredPoint::ColoredPoint() = default;

ColoredPoint::ColoredPoint(int x, int y, Color color)
    : Point(x, y), color(color) {}

std::string ColoredPoint::ToString() const {
  return absl::Substitute("($0, $1, $2)", ColorToString(color), x, y);
}

}  // namespace tachyon::base::test
