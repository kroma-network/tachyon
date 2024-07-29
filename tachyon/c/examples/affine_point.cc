#include <stdio.h>

#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2.h"

int main() {
  // Initialize the G1 and G2 curves
  tachyon_bn254_g1_init();
  tachyon_bn254_g2_init();

  // Generate random points on G1 and G2 curves
  tachyon_bn254_g1_affine g1_random_point = tachyon_bn254_g1_affine_random();
  tachyon_bn254_g2_affine g2_random_point = tachyon_bn254_g2_affine_random();

  // Generate the generator points of G1 and G2 curves
  tachyon_bn254_g1_affine g1_generator_point =
      tachyon_bn254_g1_affine_generator();
  tachyon_bn254_g2_affine g2_generator_point =
      tachyon_bn254_g2_affine_generator();

  // Perform addition on G1 and G2 curves
  [[maybe_unused]] tachyon_bn254_g1_jacobian g1_sum =
      tachyon_bn254_g1_affine_add(&g1_random_point, &g1_generator_point);
  [[maybe_unused]] tachyon_bn254_g2_jacobian g2_sum =
      tachyon_bn254_g2_affine_add(&g2_random_point, &g2_generator_point);

  // Perform negation on G1 and G2 curves
  [[maybe_unused]] tachyon_bn254_g1_affine g1_neg =
      tachyon_bn254_g1_affine_neg(&g1_random_point);
  [[maybe_unused]] tachyon_bn254_g2_affine g2_neg =
      tachyon_bn254_g2_affine_neg(&g2_random_point);

  // Double the points on G1 and G2 curves
  [[maybe_unused]] tachyon_bn254_g1_jacobian g1_doubled =
      tachyon_bn254_g1_affine_dbl(&g1_random_point);
  [[maybe_unused]] tachyon_bn254_g2_jacobian g2_doubled =
      tachyon_bn254_g2_affine_dbl(&g2_random_point);

  printf("Affine point operations on G1 and G2 curves completed.\n");

  return 0;
}
