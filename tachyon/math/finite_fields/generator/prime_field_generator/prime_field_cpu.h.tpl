// clang-format off
#include "%{config_header_path}"

%{if kUseAsm}
#if ARCH_CPU_X86_64
#include "%{prime_field_x86_hdr}"
#else
%{endif kUseAsm}
#include "tachyon/math/finite_fields/prime_field_fallback.h"
%{if kUseAsm}
#endif
%{endif kUseAsm}

namespace %{namespace} {

using %{class} = PrimeField<%{class}Config>;

}  // namespace %{namespace}
// clang-format on
