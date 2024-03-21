// clang-format off
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fq%{base_field_degree}.h"

struct tachyon_%{type}_fq%{degree} {
  tachyon_%{type}_fq%{base_field_degree} c0;
  tachyon_%{type}_fq%{base_field_degree} c1;
%{if IsCubicExtension}
  tachyon_%{type}_fq%{base_field_degree} c2;
%{endif IsCubicExtension}
};

%{extern_c_front}

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_zero();

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_one();

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_random();

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_dbl(const tachyon_%{type}_fq%{degree}* a);

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_neg(const tachyon_%{type}_fq%{degree}* a);

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_sqr(const tachyon_%{type}_fq%{degree}* a);

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_inv(const tachyon_%{type}_fq%{degree}* a);

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_add(const tachyon_%{type}_fq%{degree}* a,
                                            const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_sub(const tachyon_%{type}_fq%{degree}* a,
                                            const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_mul(const tachyon_%{type}_fq%{degree}* a,
                                            const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_div(const tachyon_%{type}_fq%{degree}* a,
                                            const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_eq(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_ne(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_gt(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_ge(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_lt(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_le(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);
// clang-format on
