#ifndef TACHYON_BASE_BINDING_TEST_COLORED_POINT_H_
#define TACHYON_BASE_BINDING_TEST_COLORED_POINT_H_

#include "tachyon/base/binding/test/color.h"
#include "tachyon/base/binding/test/point.h"

namespace tachyon::base::test {

struct ColoredPoint : public Point {
  Color color;

  ColoredPoint();
  ColoredPoint(int x, int y, Color color);

  std::string ToString() const override;
};

}  // namespace tachyon::base::test

#endif  // TACHYON_BASE_BINDING_TEST_COLORED_POINT_H_
