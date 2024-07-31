#include <stdio.h>

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"

int main() {
  // Initialize prime field elements
  tachyon_bn254_fr a = tachyon_bn254_fr_random();
  tachyon_bn254_fr b = tachyon_bn254_fr_random();
  printf("Random prime field elements a and b have been generated.\n");

  // Addition
  [[maybe_unused]] tachyon_bn254_fr sum = tachyon_bn254_fr_add(&a, &b);
  printf("a + b computed.\n");

  // Subtraction
  [[maybe_unused]] tachyon_bn254_fr difference = tachyon_bn254_fr_sub(&a, &b);
  printf("a - b computed.\n");

  // Multiplication
  [[maybe_unused]] tachyon_bn254_fr product = tachyon_bn254_fr_mul(&a, &b);
  printf("a * b computed.\n");

  // Division
  [[maybe_unused]] tachyon_bn254_fr quotient = tachyon_bn254_fr_div(&a, &b);
  printf("a / b computed.\n");

  // Negation
  [[maybe_unused]] tachyon_bn254_fr negation = tachyon_bn254_fr_neg(&a);
  printf("-a computed.\n");

  // Doubling
  [[maybe_unused]] tachyon_bn254_fr doubled = tachyon_bn254_fr_dbl(&a);
  printf("2a computed.\n");

  // Squaring
  [[maybe_unused]] tachyon_bn254_fr squared = tachyon_bn254_fr_sqr(&a);
  printf("a^2 computed.\n");

  // Inversion
  [[maybe_unused]] tachyon_bn254_fr inverse = tachyon_bn254_fr_inv(&a);
  printf("a^-1 computed.\n");

  // Equality check
  int isEqual = tachyon_bn254_fr_eq(&a, &b);
  printf("Checking if a == b: %s\n", isEqual ? "true" : "false");

  // Inequality check
  int isNotEqual = tachyon_bn254_fr_ne(&a, &b);
  printf("Checking if a != b: %s\n", isNotEqual ? "true" : "false");

  return 0;
}
