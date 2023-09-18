#include "tachyon/base/binding/test/rect.h"

namespace tachyon::base::test {

Rect::Rect() = default;

Rect::Rect(const Point& top_left, const Point& bottom_right)
    : top_left(top_left), bottom_right(bottom_right) {}

}  // namespace tachyon::base::test
