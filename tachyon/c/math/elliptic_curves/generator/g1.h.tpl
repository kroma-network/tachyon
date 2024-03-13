// clang-format off
#include "tachyon/c/export.h"
#include "%{header_path}"

struct __attribute__((aligned(32))) tachyon_%{type}_g1_affine {
  tachyon_%{type}_fq x;
  tachyon_%{type}_fq y;
  // needs to occupy 32 byte
  // NOTE(chokobole): See LimbsAlignment() in tachyon/math/base/big_int.h
  bool infinity;
};

struct tachyon_%{type}_g1_projective {
  tachyon_%{type}_fq x;
  tachyon_%{type}_fq y;
  tachyon_%{type}_fq z;
};

struct tachyon_%{type}_g1_jacobian {
  tachyon_%{type}_fq x;
  tachyon_%{type}_fq y;
  tachyon_%{type}_fq z;
};

struct tachyon_%{type}_g1_xyzz {
  tachyon_%{type}_fq x;
  tachyon_%{type}_fq y;
  tachyon_%{type}_fq zz;
  tachyon_%{type}_fq zzz;
};

struct tachyon_%{type}_g1_point2 {
  tachyon_%{type}_fq x;
  tachyon_%{type}_fq y;
};

struct tachyon_%{type}_g1_point3 {
  tachyon_%{type}_fq x;
  tachyon_%{type}_fq y;
  tachyon_%{type}_fq z;
};

struct tachyon_%{type}_g1_point4 {
  tachyon_%{type}_fq x;
  tachyon_%{type}_fq y;
  tachyon_%{type}_fq z;
  tachyon_%{type}_fq w;
};

%{extern_c_front}

TACHYON_C_EXPORT void tachyon_%{type}_g1_init();

TACHYON_C_EXPORT tachyon_%{type}_g1_affine tachyon_%{type}_g1_affine_zero();

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_zero();

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_zero();

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_zero();

TACHYON_C_EXPORT tachyon_%{type}_g1_affine tachyon_%{type}_g1_affine_generator();

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_generator();

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_generator();

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_generator();

TACHYON_C_EXPORT tachyon_%{type}_g1_affine tachyon_%{type}_g1_affine_random();

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_random();

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_random();

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_random();

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_affine_add(const tachyon_%{type}_g1_affine* a, const tachyon_%{type}_g1_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_add(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_projective* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_add_mixed(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_add(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_jacobian* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_add_mixed(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_add(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_xyzz* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_add_mixed(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_affine_sub(const tachyon_%{type}_g1_affine* a, const tachyon_%{type}_g1_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_sub(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_projective* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_sub_mixed(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_sub(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_jacobian* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_sub_mixed(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_sub(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_xyzz* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_sub_mixed(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_affine* b);

TACHYON_C_EXPORT tachyon_%{type}_g1_affine tachyon_%{type}_g1_affine_neg(const tachyon_%{type}_g1_affine* a);

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_neg(const tachyon_%{type}_g1_projective* a);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_neg(const tachyon_%{type}_g1_jacobian* a);

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_neg(const tachyon_%{type}_g1_xyzz* a);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_affine_dbl(const tachyon_%{type}_g1_affine* a);

TACHYON_C_EXPORT tachyon_%{type}_g1_projective tachyon_%{type}_g1_projective_dbl(const tachyon_%{type}_g1_projective* a);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian tachyon_%{type}_g1_jacobian_dbl(const tachyon_%{type}_g1_jacobian* a);

TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz tachyon_%{type}_g1_xyzz_dbl(const tachyon_%{type}_g1_xyzz* a);

bool TACHYON_C_EXPORT tachyon_%{type}_g1_affine_eq(const tachyon_%{type}_g1_affine* a, const tachyon_%{type}_g1_affine* b);

bool TACHYON_C_EXPORT tachyon_%{type}_g1_projective_eq(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_projective* b);

bool TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian_eq(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_jacobian* b);

bool TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz_eq(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_xyzz* b);

bool TACHYON_C_EXPORT tachyon_%{type}_g1_affine_ne(const tachyon_%{type}_g1_affine* a, const tachyon_%{type}_g1_affine* b);

bool TACHYON_C_EXPORT tachyon_%{type}_g1_projective_ne(const tachyon_%{type}_g1_projective* a, const tachyon_%{type}_g1_projective* b);

bool TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian_ne(const tachyon_%{type}_g1_jacobian* a, const tachyon_%{type}_g1_jacobian* b);

bool TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz_ne(const tachyon_%{type}_g1_xyzz* a, const tachyon_%{type}_g1_xyzz* b);
// clang-format on
