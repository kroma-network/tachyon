#ifndef TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_H_
#define TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_H_

#include <ostream>

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_fq& fq);

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_fr& fr);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_affine& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_projective& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_jacobian& point);

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_g1_xyzz& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_affine& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_projective& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_point2& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_point3& point);

bool operator==(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b);
bool operator!=(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b);
bool operator<(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b);
bool operator<=(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b);
bool operator>(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b);
bool operator>=(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b);

bool operator==(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b);
bool operator!=(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b);
bool operator<(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b);
bool operator<=(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b);
bool operator>(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b);
bool operator>=(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b);

bool operator==(const tachyon_bn254_g1_affine& a,
                const tachyon_bn254_g1_affine& b);
bool operator!=(const tachyon_bn254_g1_affine& a,
                const tachyon_bn254_g1_affine& b);

bool operator==(const tachyon_bn254_g1_projective& a,
                const tachyon_bn254_g1_projective& b);
bool operator!=(const tachyon_bn254_g1_projective& a,
                const tachyon_bn254_g1_projective& b);

bool operator==(const tachyon_bn254_g1_jacobian& a,
                const tachyon_bn254_g1_jacobian& b);
bool operator!=(const tachyon_bn254_g1_jacobian& a,
                const tachyon_bn254_g1_jacobian& b);

bool operator==(const tachyon_bn254_g1_xyzz& a, const tachyon_bn254_g1_xyzz& b);
bool operator!=(const tachyon_bn254_g1_xyzz& a, const tachyon_bn254_g1_xyzz& b);

bool operator==(const tachyon_bn254_g1_point2& a,
                const tachyon_bn254_g1_point2& b);
bool operator!=(const tachyon_bn254_g1_point2& a,
                const tachyon_bn254_g1_point2& b);

bool operator==(const tachyon_bn254_g1_point3& a,
                const tachyon_bn254_g1_point3& b);
bool operator!=(const tachyon_bn254_g1_point3& a,
                const tachyon_bn254_g1_point3& b);

#endif  // TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_H_
