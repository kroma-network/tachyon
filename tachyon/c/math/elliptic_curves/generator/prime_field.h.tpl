// clang-format off
#include <stdint.h>

#include "tachyon/c/export.h"

struct tachyon_%{type}_%{suffix} {
  uint64_t limbs[%{limb_nums}];
};

%{extern_c_front}

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_zero();

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_one();

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_random();

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_add(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_sub(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_mul(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_div(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_neg(const tachyon_%{type}_%{suffix}* a);

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_dbl(const tachyon_%{type}_%{suffix}* a);

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_sqr(const tachyon_%{type}_%{suffix}* a);

TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_inv(const tachyon_%{type}_%{suffix}* a);

TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_eq(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_ne(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_gt(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_ge(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_lt(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_le(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

// clang-format on
