#include <stdio.h>

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq12.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq2.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq6.h"

int main() {
  // Initialize extension field elements
  tachyon_bn254_fq2 fq2_element = tachyon_bn254_fq2_random();
  tachyon_bn254_fq6 fq6_element = tachyon_bn254_fq6_random();
  tachyon_bn254_fq12 fq12_element = tachyon_bn254_fq12_random();

  printf("Random Fq2, Fq6, and Fq12 elements have been generated.\n");

  // Perform operations on Fq2 element
  tachyon_bn254_fq2 fq2_negated = tachyon_bn254_fq2_neg(&fq2_element);
  tachyon_bn254_fq2 fq2_doubled = tachyon_bn254_fq2_dbl(&fq2_element);
  tachyon_bn254_fq2 fq2_squared = tachyon_bn254_fq2_sqr(&fq2_element);
  tachyon_bn254_fq2 fq2_inverted = tachyon_bn254_fq2_inv(&fq2_element);

  printf("Operations on Fq2 element completed.\n");

  // Perform operations on Fq6 element
  tachyon_bn254_fq6 fq6_negated = tachyon_bn254_fq6_neg(&fq6_element);
  tachyon_bn254_fq6 fq6_doubled = tachyon_bn254_fq6_dbl(&fq6_element);
  tachyon_bn254_fq6 fq6_squared = tachyon_bn254_fq6_sqr(&fq6_element);
  tachyon_bn254_fq6 fq6_inverted = tachyon_bn254_fq6_inv(&fq6_element);

  printf("Operations on Fq6 element completed.\n");

  // Perform operations on Fq12 element
  tachyon_bn254_fq12 fq12_negated = tachyon_bn254_fq12_neg(&fq12_element);
  tachyon_bn254_fq12 fq12_doubled = tachyon_bn254_fq12_dbl(&fq12_element);
  tachyon_bn254_fq12 fq12_squared = tachyon_bn254_fq12_sqr(&fq12_element);
  tachyon_bn254_fq12 fq12_inverted = tachyon_bn254_fq12_inv(&fq12_element);

  printf("Operations on Fq12 element completed.\n");

  return 0;
}
