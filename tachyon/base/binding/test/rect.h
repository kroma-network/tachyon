#ifndef TACHYON_BASE_BINDING_TEST_RECT_H_
#define TACHYON_BASE_BINDING_TEST_RECT_H_

#include "tachyon/base/binding/test/point.h"

namespace tachyon::base::test {

struct Rect {
  Point top_left;
  Point bottom_right;

  Rect();
  Rect(const Point& top_left, const Point& bottom_right);

  Point& GetTopLeft() { return top_left; }
  const Point& GetConstTopLeft() const { return top_left; }

  Point& GetBottomRight() { return bottom_right; }
  const Point& GetConstBottomRight() const { return bottom_right; }
};

}  // namespace tachyon::base::test

#endif  // TACHYON_BASE_BINDING_TEST_RECT_H_
