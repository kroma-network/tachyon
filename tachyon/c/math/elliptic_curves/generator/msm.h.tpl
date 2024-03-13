// clang-format off
#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fr.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1.h"

typedef struct tachyon_%{type}_g1_msm* tachyon_%{type}_g1_msm_ptr;

%{extern_c_front}

TACHYON_C_EXPORT tachyon_%{type}_g1_msm_ptr tachyon_%{type}_g1_create_msm(uint8_t degree);

TACHYON_C_EXPORT void tachyon_%{type}_g1_destroy_msm(tachyon_%{type}_g1_msm_ptr ptr);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian* tachyon_%{type}_g1_point2_msm(
    tachyon_%{type}_g1_msm_ptr ptr, const tachyon_%{type}_g1_point2* bases,
    const tachyon_%{type}_fr* scalars, size_t size);

TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian* tachyon_%{type}_g1_affine_msm(
    tachyon_%{type}_g1_msm_ptr ptr, const tachyon_%{type}_g1_affine* bases,
    const tachyon_%{type}_fr* scalars, size_t size);
// clang-format off
