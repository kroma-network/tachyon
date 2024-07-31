#include <stdio.h>

#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2.h"

int main() {
  // Initialize G1 and G2 curves
  tachyon_bn254_g1_init();
  tachyon_bn254_g2_init();

  // Generate random projective points on G1 and G2 curves
  tachyon_bn254_g1_projective g1_random_point =
      tachyon_bn254_g1_projective_random();
  tachyon_bn254_g2_projective g2_random_point =
      tachyon_bn254_g2_projective_random();

  // Generate generator projective points of G1 and G2 curves
  tachyon_bn254_g1_projective g1_generator_point =
      tachyon_bn254_g1_projective_generator();
  tachyon_bn254_g2_projective g2_generator_point =
      tachyon_bn254_g2_projective_generator();

  // Perform projective point addition on G1 and G2 curves
  [[maybe_unused]] tachyon_bn254_g1_projective g1_sum =
      tachyon_bn254_g1_projective_add(&g1_random_point, &g1_generator_point);
  [[maybe_unused]] tachyon_bn254_g2_projective g2_sum =
      tachyon_bn254_g2_projective_add(&g2_random_point, &g2_generator_point);

  // Double the projective points on G1 and G2 curves
  [[maybe_unused]] tachyon_bn254_g1_projective g1_doubled =
      tachyon_bn254_g1_projective_dbl(&g1_random_point);
  [[maybe_unused]] tachyon_bn254_g2_projective g2_doubled =
      tachyon_bn254_g2_projective_dbl(&g2_random_point);

  printf("Projective point operations on G1 and G2 curves completed.\n");

  return 0;
}
