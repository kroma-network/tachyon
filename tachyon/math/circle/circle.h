#ifndef TACHYON_MATH_CIRCLE_CIRCLE_H_
#define TACHYON_MATH_CIRCLE_CIRCLE_H_

#include "tachyon/math/circle/circle_point.h"
#include "tachyon/math/circle/circle_traits_forward.h"

namespace tachyon::math {

// Config for Unit Circle.
// This config represents x² + y² = 1.
template <typename CircleConfig>
class Circle {
 public:
  using Config = CircleConfig;

  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using Point = typename CircleTraits<Config>::PointTy;

  static void Init() {
    BaseField::Init();
    ScalarField::Init();

    Config::Init();
  }

  constexpr static bool IsOnCircle(const Point& point) {
    return (point.x().Square() + point.y().Square()).IsOne();
  }
};

template <typename Config>
struct CircleTraits {
  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using PointTy = CirclePoint<Circle<Config>>;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_CIRCLE_CIRCLE_H_
