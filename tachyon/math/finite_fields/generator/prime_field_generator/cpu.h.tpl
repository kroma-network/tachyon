// clang-format off
#include "%{config_header_path}"

%{if kIsSmallField}
#include "tachyon/math/finite_fields/small_prime_field_generic.h"
%{endif kIsSmallField}
%{if !kIsSmallField}
#if ARCH_CPU_X86_64
#include "%{prime_field_x86_hdr}"
#else
#include "tachyon/math/finite_fields/prime_field_generic.h"
#endif
%{endif !kIsSmallField}

namespace %{namespace} {

using %{class} = PrimeField<%{class}Config>;

}  // namespace %{namespace}
// clang-format on
