#ifndef TACHYON_BASE_BINDING_TEST_POINT_H_
#define TACHYON_BASE_BINDING_TEST_POINT_H_

#include <math.h>

#include <string>

#include "absl/strings/substitute.h"

namespace tachyon::base::test {

struct Point {
  static int s_dimension;

  int x = 0;
  int y = 0;

  Point();
  Point(int x, int y);
  virtual ~Point();

  static double Distance(const Point& p1, const Point& p2);

  static void SetDimension(int dimension) { s_dimension = dimension; }
  static int GetDimension() { return s_dimension; }

  int GetX() const { return x; }
  int GetY() const { return y; }
  void SetX(int x) { this->x = x; }
  void SetY(int y) { this->y = y; }

  virtual std::string ToString() const;
};

}  // namespace tachyon::base::test

#endif  // TACHYON_BASE_BINDING_TEST_POINT_H_
