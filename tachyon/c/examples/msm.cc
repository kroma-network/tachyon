#include "tachyon/c/math/elliptic_curves/bn/bn254/msm.h"

#include <stdio.h>
#include <stdlib.h>

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"

int main() {
  // Initialize curve and MSM context
  tachyon_bn254_g1_init();

  static tachyon_bn254_g1_msm_ptr msm = tachyon_bn254_g1_create_msm(32);

  const size_t kMSMSize = 32;

  // Example: Scalar values to multiply with G1 points
  tachyon_bn254_fr scalars[kMSMSize];
  for (size_t i = 0; i < kMSMSize; i++) {
    scalars[i] = tachyon_bn254_fr_random();
  }

  // Example: G1 points
  tachyon_bn254_g1_affine points[kMSMSize];
  for (size_t i = 0; i < kMSMSize; i++) {
    points[i] = tachyon_bn254_g1_affine_random();
  }

  // Perform the MSM operation
  tachyon_bn254_g1_jacobian* result =
      tachyon_bn254_g1_affine_msm(msm, points, scalars, 32);

  return 0;
}
