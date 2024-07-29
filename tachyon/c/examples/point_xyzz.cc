#include <stdio.h>

#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2.h"

int main() {
  // Initialize the curves
  tachyon_bn254_g1_init();
  tachyon_bn254_g2_init();

  // Generate random `PointXYZZ` points on G1 and G2
  tachyon_bn254_g1_xyzz g1_random_point = tachyon_bn254_g1_xyzz_random();
  tachyon_bn254_g2_xyzz g2_random_point = tachyon_bn254_g2_xyzz_random();

  // Demonstrate addition with a generated point
  [[maybe_unused]] tachyon_bn254_g1_xyzz g1_point_sum =
      tachyon_bn254_g1_xyzz_add(
          &g1_random_point,
          &g1_random_point);  // Adding to itself as an example
  [[maybe_unused]] tachyon_bn254_g2_xyzz g2_point_sum =
      tachyon_bn254_g2_xyzz_add(&g2_random_point, &g2_random_point);

  // Demonstrate doubling
  [[maybe_unused]] tachyon_bn254_g1_xyzz g1_doubled =
      tachyon_bn254_g1_xyzz_dbl(&g1_random_point);
  [[maybe_unused]] tachyon_bn254_g2_xyzz g2_doubled =
      tachyon_bn254_g2_xyzz_dbl(&g2_random_point);

  // Demonstrate negation
  tachyon_bn254_g1_xyzz g1_negated =
      tachyon_bn254_g1_xyzz_neg(&g1_random_point);
  tachyon_bn254_g2_xyzz g2_negated =
      tachyon_bn254_g2_xyzz_neg(&g2_random_point);

  // Demonstrate subtraction (using negation + addition for demonstration)
  [[maybe_unused]] tachyon_bn254_g1_xyzz g1_sub_result =
      tachyon_bn254_g1_xyzz_add(&g1_random_point, &g1_negated);
  [[maybe_unused]] tachyon_bn254_g2_xyzz g2_sub_result =
      tachyon_bn254_g2_xyzz_add(&g2_random_point, &g2_negated);

  printf("Point XYZZ operations on G1 and G2 curves completed.\n");

  return 0;
}
