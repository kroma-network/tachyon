// Copyright 2024 StarkWare Industries Ltd
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE-APACHE.stwo

#ifndef TACHYON_MATH_CIRCLE_CIRCLE_POINT_FORWARD_H_
#define TACHYON_MATH_CIRCLE_CIRCLE_POINT_FORWARD_H_

namespace tachyon::math {

template <typename Circle>
class CirclePoint;

template <typename ScalarField, typename Circle>
CirclePoint<Circle> operator*(const ScalarField& v,
                              const CirclePoint<Circle>& point) {
  return point * v;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_CIRCLE_CIRCLE_POINT_FORWARD_H_
