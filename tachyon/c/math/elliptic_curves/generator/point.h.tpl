// clang-format off
#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{base_field}.h"

struct __attribute__((aligned(32))) tachyon_%{type}_%{g1_or_g2}_affine {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  // needs to occupy 32 byte
  // NOTE(chokobole): See LimbsAlignment() in tachyon/math/base/big_int.h
  bool infinity;
};

struct tachyon_%{type}_%{g1_or_g2}_projective {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} z;
};

struct tachyon_%{type}_%{g1_or_g2}_jacobian {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} z;
};

struct tachyon_%{type}_%{g1_or_g2}_xyzz {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} zz;
  tachyon_%{type}_%{base_field} zzz;
};

struct tachyon_%{type}_%{g1_or_g2}_point2 {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
};

struct tachyon_%{type}_%{g1_or_g2}_point3 {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} z;
};

struct tachyon_%{type}_%{g1_or_g2}_point4 {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} z;
  tachyon_%{type}_%{base_field} w;
};

%{extern_c_front}

TACHYON_C_EXPORT void tachyon_%{type}_%{g1_or_g2}_init();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_zero();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_zero();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_zero();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_zero();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_generator();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_generator();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_generator();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_generator();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_random();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_random();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_random();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_random();

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_add(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_add(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_add_mixed(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_add(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_add_mixed(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_add(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_add_mixed(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_sub(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_sub(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_sub(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_sub(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_neg(const tachyon_%{type}_%{g1_or_g2}_affine* a);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_neg(const tachyon_%{type}_%{g1_or_g2}_projective* a);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_neg(const tachyon_%{type}_%{g1_or_g2}_jacobian* a);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_neg(const tachyon_%{type}_%{g1_or_g2}_xyzz* a);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_dbl(const tachyon_%{type}_%{g1_or_g2}_affine* a);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_dbl(const tachyon_%{type}_%{g1_or_g2}_projective* a);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_dbl(const tachyon_%{type}_%{g1_or_g2}_jacobian* a);

TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_dbl(const tachyon_%{type}_%{g1_or_g2}_xyzz* a);

bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine_eq(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective_eq(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b);

bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian_eq(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b);

bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz_eq(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b);

bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine_ne(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective_ne(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b);

bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian_ne(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b);

bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz_ne(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b);
// clang-format on
